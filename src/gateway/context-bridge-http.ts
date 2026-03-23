import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import { agentCommandFromIngress } from "../commands/agent.js";
import { loadConfig } from "../config/config.js";
import { callGateway } from "./call.js";
import type { AuthRateLimiter } from "./auth-rate-limit.js";
import { authorizeHttpGatewayConnect, type ResolvedGatewayAuth } from "./auth.js";
import {
  readJsonBodyOrError,
  sendGatewayAuthFailure,
  sendInvalidRequest,
  sendJson,
  sendMethodNotAllowed,
} from "./http-common.js";
import { getBearerToken } from "./http-utils.js";

const DEFAULT_BODY_BYTES = 2 * 1024 * 1024;
const MAX_CONTEXT_SOURCES = 8;
const MAX_SOURCE_LIMIT = 200;
const DEFAULT_SOURCE_LIMIT = 60;
const DEFAULT_REPLY_SCAN_LIMIT = 50;

type ContextSource = {
  sessionId?: unknown;
  sessionKey?: unknown;
  instanceId?: unknown;
  limit?: unknown;
};

type ContextBridgeBody = {
  message?: unknown;
  sessionId?: unknown;
  sessionKey?: unknown;
  sourceSessions?: unknown;
};

type ResolvedSource = {
  instanceId?: string;
  sessionId?: string;
  sessionKey?: string;
  limit: number;
};

type SourceHistory = {
  instanceId?: string;
  sessionId?: string;
  sessionKey: string;
  messages: unknown[];
};

function normalizeOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizeSourceLimit(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_SOURCE_LIMIT;
  }
  return Math.max(1, Math.min(MAX_SOURCE_LIMIT, Math.floor(value)));
}

function resolveLatestAssistantText(messages: unknown[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const candidate = messages[i];
    if (!candidate || typeof candidate !== "object") {
      continue;
    }
    const role = (candidate as { role?: unknown }).role;
    if (role !== "assistant") {
      continue;
    }
    const content = (candidate as { content?: unknown }).content;
    if (typeof content === "string" && content.trim()) {
      return content.trim();
    }
    if (Array.isArray(content)) {
      const parts = content
        .map((entry) => {
          if (!entry || typeof entry !== "object") {
            return "";
          }
          const text = (entry as { text?: unknown }).text;
          return typeof text === "string" ? text : "";
        })
        .filter(Boolean);
      const combined = parts.join("\n").trim();
      if (combined) {
        return combined;
      }
    }
  }
  return undefined;
}

async function resolveSessionKeyFromId(sessionId: string): Promise<string> {
  const resolved = await callGateway<{ key?: string }>({
    method: "sessions.resolve",
    params: {
      sessionId,
      includeGlobal: true,
      includeUnknown: true,
    },
  });
  const key = normalizeOptionalString(resolved?.key);
  if (!key) {
    throw new Error(`Session not found for sessionId: ${sessionId}`);
  }
  return key;
}

async function fetchLocalHistory(source: ResolvedSource): Promise<SourceHistory> {
  const sessionKey = source.sessionKey ?? (source.sessionId ? await resolveSessionKeyFromId(source.sessionId) : undefined);
  if (!sessionKey) {
    throw new Error("source requires sessionKey or sessionId");
  }
  const history = await callGateway<{ sessionId?: string; messages?: unknown[] }>({
    method: "chat.history",
    params: { sessionKey, limit: source.limit },
  });
  return {
    sessionId: normalizeOptionalString(history?.sessionId) ?? source.sessionId,
    sessionKey,
    messages: Array.isArray(history?.messages) ? history.messages : [],
  };
}

async function fetchRemoteHistory(source: ResolvedSource): Promise<SourceHistory> {
  if (!source.instanceId) {
    throw new Error("instanceId required for remote history fetch");
  }
  const invoke = await callGateway<{
    payload?: {
      sessionId?: string;
      sessionKey?: string;
      messages?: unknown[];
    };
  }>({
    method: "node.invoke",
    params: {
      nodeId: source.instanceId,
      command: "context.history.fetch",
      params: {
        sessionId: source.sessionId,
        sessionKey: source.sessionKey,
        limit: source.limit,
      },
      timeoutMs: 20_000,
      idempotencyKey: randomUUID(),
    },
  });
  const payload = invoke?.payload;
  const sessionKey = normalizeOptionalString(payload?.sessionKey);
  if (!sessionKey) {
    throw new Error(`remote node ${source.instanceId} did not return sessionKey`);
  }
  return {
    instanceId: source.instanceId,
    sessionId: normalizeOptionalString(payload?.sessionId) ?? source.sessionId,
    sessionKey,
    messages: Array.isArray(payload?.messages) ? payload.messages : [],
  };
}

function buildContextPrompt(targetSessionKey: string, targetSessionId: string, sources: SourceHistory[]): string {
  const serialized = sources.map((source, index) => {
    const instanceLine = source.instanceId ? `instanceId=${source.instanceId}` : "instanceId=local";
    return [
      `Source ${index + 1}: ${instanceLine}, sessionKey=${source.sessionKey}, sessionId=${source.sessionId ?? "unknown"}`,
      JSON.stringify(source.messages),
    ].join("\n");
  });
  return [
    "Context Bridge metadata:",
    `currentSessionKey=${targetSessionKey}`,
    `currentSessionId=${targetSessionId}`,
    "",
    "Cross-session context (JSON transcript fragments):",
    ...serialized,
    "",
    "When asked, you may report currentSessionId/currentSessionKey and source session identifiers.",
  ].join("\n");
}

function normalizeSources(value: unknown): ResolvedSource[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .slice(0, MAX_CONTEXT_SOURCES)
    .map((entry) => {
      const source = (entry ?? {}) as ContextSource;
      return {
        sessionId: normalizeOptionalString(source.sessionId),
        sessionKey: normalizeOptionalString(source.sessionKey),
        instanceId: normalizeOptionalString(source.instanceId),
        limit: normalizeSourceLimit(source.limit),
      } satisfies ResolvedSource;
    })
    .filter((entry) => entry.sessionId || entry.sessionKey);
}

export async function handleContextBridgeHttpRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts: {
    auth: ResolvedGatewayAuth;
    maxBodyBytes?: number;
    trustedProxies?: string[];
    allowRealIpFallback?: boolean;
    rateLimiter?: AuthRateLimiter;
  },
): Promise<boolean> {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
  if (url.pathname !== "/channels/context-bridge/send") {
    return false;
  }
  if (req.method !== "POST") {
    sendMethodNotAllowed(res, "POST");
    return true;
  }

  const cfg = loadConfig();
  const token = getBearerToken(req);
  const authResult = await authorizeHttpGatewayConnect({
    auth: opts.auth,
    connectAuth: token ? { token, password: token } : null,
    req,
    trustedProxies: opts.trustedProxies ?? cfg.gateway?.trustedProxies,
    allowRealIpFallback: opts.allowRealIpFallback ?? cfg.gateway?.allowRealIpFallback,
    rateLimiter: opts.rateLimiter,
  });
  if (!authResult.ok) {
    sendGatewayAuthFailure(res, authResult);
    return true;
  }

  const bodyUnknown = await readJsonBodyOrError(req, res, opts.maxBodyBytes ?? DEFAULT_BODY_BYTES);
  if (bodyUnknown === undefined) {
    return true;
  }
  const body = (bodyUnknown ?? {}) as ContextBridgeBody;
  const message = normalizeOptionalString(body.message);
  if (!message) {
    sendInvalidRequest(res, "context bridge requires body.message");
    return true;
  }

  const targetSessionId = normalizeOptionalString(body.sessionId) ?? randomUUID();
  const targetSessionKey =
    normalizeOptionalString(body.sessionKey) ?? `agent:main:contextbridge:run:${targetSessionId}`;

  const sources = normalizeSources(body.sourceSessions);
  const sourceHistories: SourceHistory[] = [];
  for (const source of sources) {
    const history = source.instanceId
      ? await fetchRemoteHistory(source)
      : await fetchLocalHistory(source);
    sourceHistories.push(history);
  }

  const runId = randomUUID();
  await agentCommandFromIngress({
    runId,
    message,
    sessionId: targetSessionId,
    sessionKey: targetSessionKey,
    thinking: "low",
    deliver: false,
    messageChannel: "webchat",
    extraSystemPrompt: buildContextPrompt(targetSessionKey, targetSessionId, sourceHistories),
    senderIsOwner: false,
    allowModelOverride: false,
  });

  const targetHistory = await callGateway<{ messages?: unknown[] }>({
    method: "chat.history",
    params: { sessionKey: targetSessionKey, limit: DEFAULT_REPLY_SCAN_LIMIT },
  });
  const messages = Array.isArray(targetHistory?.messages) ? targetHistory.messages : [];
  const reply = resolveLatestAssistantText(messages) ?? "";

  sendJson(res, 200, {
    ok: true,
    sessionKey: targetSessionKey,
    sessionId: targetSessionId,
    runId,
    reply,
    contextSourcesUsed: sourceHistories.map((source) => ({
      instanceId: source.instanceId ?? null,
      sessionId: source.sessionId ?? null,
      sessionKey: source.sessionKey,
      messageCount: source.messages.length,
    })),
  });
  return true;
}
