/**
 * Per-user API key via HTTP-only cookie.
 *
 * When the UI is deployed without TENSORZERO_API_KEY but the gateway
 * requires auth, users can enter a key in the browser. That key is
 * stored in an HTTP-only cookie so each browser has its own auth state.
 *
 * AsyncLocalStorage makes the current request available to any function
 * in the call chain without threading `request` through every parameter.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { getEnv } from "./env.server";

const COOKIE_NAME = "tz_gateway_key";
const MAX_AGE_SECONDS = 2592000; // 30 days

const requestStore = new AsyncLocalStorage<Request>();

/** Wrap a handler so getEffectiveApiKey() can read the request's cookies. */
export function runWithRequest<T>(request: Request, fn: () => T): T {
  return requestStore.run(request, fn);
}

function parseCookieValue(
  cookieHeader: string,
  name: string,
): string | undefined {
  const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${name}=([^;]*)`));
  if (!match) return undefined;
  const decoded = decodeURIComponent(match[1]);
  return decoded || undefined;
}

/** Read API key from the current request's HTTP-only cookie. */
export function getApiKeyFromRequest(request: Request): string | undefined {
  const cookieHeader = request.headers.get("cookie");
  if (!cookieHeader) return undefined;
  return parseCookieValue(cookieHeader, COOKIE_NAME);
}

/** Returns the env var if set, otherwise the cookie value from the current request. */
export function getEffectiveApiKey(): string | undefined {
  const envKey = getEnv().TENSORZERO_API_KEY;
  if (envKey) return envKey;

  const request = requestStore.getStore();
  if (!request) return undefined;
  return getApiKeyFromRequest(request);
}

/** Build a Set-Cookie header value to store the API key. */
export function buildApiKeyCookie(apiKey: string): string {
  const parts = [
    `${COOKIE_NAME}=${encodeURIComponent(apiKey)}`,
    "HttpOnly",
    "SameSite=Strict",
    "Path=/",
    `Max-Age=${MAX_AGE_SECONDS}`,
  ];

  if (!isLocalhost()) {
    parts.push("Secure");
  }

  return parts.join("; ");
}

/** Build a Set-Cookie header value to clear the cookie. */
export function buildClearApiKeyCookie(): string {
  const parts = [
    `${COOKIE_NAME}=`,
    "HttpOnly",
    "SameSite=Strict",
    "Path=/",
    "Max-Age=0",
  ];

  if (!isLocalhost()) {
    parts.push("Secure");
  }

  return parts.join("; ");
}

function isLocalhost(): boolean {
  const url = getEnv().TENSORZERO_GATEWAY_URL;
  try {
    const hostname = new URL(url).hostname;
    return hostname === "localhost" || hostname === "127.0.0.1";
  } catch {
    return false;
  }
}
