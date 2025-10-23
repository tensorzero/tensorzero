/**
 * Read-Only Mode Server Utilities
 *
 * Provides middleware for protecting routes from write operations when
 * the application is in read-only mode.
 *
 * @module read-only.server
 */

import { data } from "react-router";
import type { Route } from "../+types/root";
import { getEnv } from "./env.server";

/**
 * Checks if the application is in read-only mode
 * @returns true if TENSORZERO_UI_READ_ONLY is set to "1"
 */
export function isReadOnlyMode(): boolean {
  return getEnv().TENSORZERO_UI_READ_ONLY;
}

/**
 * Middleware function to protect routes from write operations in read-only mode.
 * Only allows GET and HEAD requests when read-only mode is enabled.
 *
 * Usage:
 * ```typescript
 * export const middleware: Route.MiddlewareFunction[] = [
 *   readOnlyMiddleware,
 * ];
 * ```
 *
 * @throws {Response} 403 Forbidden response if in read-only mode and method is not GET or HEAD
 */
export const readOnlyMiddleware: Route.MiddlewareFunction = async ({
  request,
}) => {
  if (isReadOnlyMode()) {
    const method = request.method.toUpperCase();
    if (method !== "GET" && method !== "HEAD") {
      throw data(
        {
          error:
            "This operation is not allowed in read-only mode. All write operations are disabled.",
        },
        { status: 403 },
      );
    }
  }
};
