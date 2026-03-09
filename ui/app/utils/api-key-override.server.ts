/**
 * Per-user API key via HTTP-only cookie.
 *
 * When the UI is deployed without TENSORZERO_API_KEY but the gateway
 * requires auth, users can enter a key in the browser. That key is
 * stored in an HTTP-only cookie so each browser has its own auth state.
 *
 * AsyncLocalStorage makes the current request available to any function
 * in the call chain without threading `request` through every parameter.
 * The middleware eagerly parses the cookie so getEffectiveApiKey() can
 * read it synchronously.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { createCookie } from "react-router";
import { getEnv } from "./env.server";

export const apiKeyCookie = createCookie("t0_gateway_key", {
  httpOnly: true,
  sameSite: "strict",
  path: "/",
  maxAge: 2592000, // 30 days
});

/** Secure flag should only be set when served over HTTPS, not plain HTTP. */
export function isSecureRequest(request: Request): boolean {
  // X-Forwarded-Proto is set by reverse proxies; may be comma-separated
  // (e.g. "https, http") in multi-proxy setups — check the first value.
  const forwarded = request.headers.get("x-forwarded-proto");
  if (forwarded) return forwarded.split(",")[0].trim() === "https";
  return new URL(request.url).protocol === "https:";
}

const apiKeyStore = new AsyncLocalStorage<string | undefined>();

/**
 * Wrap a handler so getEffectiveApiKey() can read the request's cookie.
 * Eagerly parses the cookie so downstream reads are synchronous.
 */
export async function runWithRequest<T>(
  request: Request,
  fn: () => T,
): Promise<T> {
  const value = await apiKeyCookie.parse(request.headers.get("cookie"));
  const apiKey = typeof value === "string" && value ? value : undefined;
  return apiKeyStore.run(apiKey, fn);
}

/** Read API key from a request's cookie (async, for use outside middleware). */
export async function getApiKeyFromRequest(
  request: Request,
): Promise<string | undefined> {
  const value = await apiKeyCookie.parse(request.headers.get("cookie"));
  return typeof value === "string" && value ? value : undefined;
}

/** Returns the env var if set, otherwise the cookie value from the current request. */
export function getEffectiveApiKey(): string | undefined {
  const envKey = getEnv().TENSORZERO_API_KEY;
  if (envKey) return envKey;
  return apiKeyStore.getStore();
}
