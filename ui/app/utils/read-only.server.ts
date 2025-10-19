/**
 * Returns true if the UI should operate in read-only mode.
 * Controlled via the environment variable TENSORZERO_UI_READ_ONLY.
 * Accepted truthy values: "1", "true" (case-insensitive).
 */
export function isReadOnlyMode(): boolean {
  const v = process.env.TENSORZERO_UI_READ_ONLY;
  if (!v) return false;
  if (v === "1") return true;
  return v.toLowerCase() === "true";
}

/**
 * Use at the top of server actions that perform write operations.
 * If read-only mode is enabled, throws an HTTP 403 response (JSON) to halt the action.
 *
 * Usage:
 *   export async function action({ request }: Route.ActionArgs) {
 *     protectAction();
 *     // ... rest of action
 *   }
 */
export function protectAction(): void {
  if (isReadOnlyMode()) {
    throw new Response(
      JSON.stringify({
        error:
          "Read-only mode is enabled. This operation is disabled in read-only mode.",
      }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
