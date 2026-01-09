/**
 * ERROR BOUNDARY ARCHITECTURE
 * ===========================
 *
 * Errors are classified into two categories:
 *
 * 1. INFRA ERRORS (gateway down, auth failed, ClickHouse unavailable):
 *    - Shown as dismissible dark modal overlay via ErrorDialog
 *    - Sidebar remains visible behind overlay
 *    - User can dismiss to browse (with degraded functionality)
 *
 * 2. PAGE ERRORS (404s, API failures, validation errors):
 *    - Shown inline within content area via PageErrorStack
 *    - Sidebar stays fully functional
 *
 * Components:
 *    - PageErrorContent: Page errors only (for routes without layout boundaries)
 *    - RootErrorBoundaryLayout: Shell with sidebar for root-level errors
 */

export { RootErrorBoundaryLayout } from "./RootErrorBoundaryLayout";
export { ErrorDialog } from "./ErrorDialog";
export { ErrorContent } from "./ErrorContent";
export { PageErrorContent } from "./PageErrorContent";
