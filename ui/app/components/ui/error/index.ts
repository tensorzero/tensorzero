/**
 * ERROR BOUNDARY ARCHITECTURE
 * ===========================
 *
 * This codebase uses a two-tier error boundary strategy:
 *
 * 1. ROOT ErrorBoundary (root.tsx):
 *    - Catches startup failures (gateway down, auth failed, ClickHouse unavailable)
 *    - Shows dark modal overlay via ErrorDialog + ErrorContent
 *    - RootErrorBoundaryLayout renders the app shell (sidebar) behind the overlay
 *      so the app feels "present but blocked" rather than completely broken
 *
 * 2. ROUTE ErrorBoundaries (e.g., datasets/layout.tsx):
 *    - Catches errors after app has loaded (API failures, validation, render errors)
 *    - Shows light card within content area - sidebar stays visible
 *    - Uses RouteErrorContent for consistent styling
 *
 * ErrorScope (App/Page) determines theming:
 *    - App: Dark overlay for app-level errors (used by ErrorContent)
 *    - Page: Light card for page-level errors (used by RouteErrorContent)
 */

export { RootErrorBoundaryLayout } from "./RootErrorBoundaryLayout";
export { ErrorDialog } from "./ErrorDialog";
export { ErrorContent } from "./ErrorContent";
export { RouteErrorContent } from "./RouteErrorContent";
