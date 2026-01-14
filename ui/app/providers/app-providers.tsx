import { ReactQueryProvider } from "./react-query";
import { GlobalToastProvider } from "./global-toast-provider";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { Toaster } from "~/components/ui/toaster";
import { ReadOnlyProvider } from "~/context/read-only";
import { AutopilotAvailableProvider } from "~/context/autopilot-available";

/**
 * Loader data fields used by AppProviders for context setup.
 * These have safe defaults (false) when not provided.
 */
export interface AppProvidersLoaderData {
  isReadOnly?: boolean;
  autopilotAvailable?: boolean;
}

interface AppProvidersProps {
  children: React.ReactNode;
  /**
   * Loader data for context providers. When provided (e.g., from useRouteLoaderData),
   * enables the sidebar to show correct read-only badge and autopilot section state.
   * Falls back to safe defaults (false) when data is unavailable.
   */
  loaderData?: AppProvidersLoaderData;
}

/**
 * Shared provider stack for the app shell.
 * Used by both the main App and ErrorAppShell to ensure consistency.
 *
 * Note: ConfigProvider is intentionally not included here yet because it requires
 * a valid config value (no safe default). Once EMPTY_CONFIG is introduced for
 * graceful degradation (PR 2a), ConfigProvider will be moved here as well.
 */
export function AppProviders({ children, loaderData }: AppProvidersProps) {
  return (
    <ReactQueryProvider>
      <GlobalToastProvider>
        <ReadOnlyProvider value={loaderData?.isReadOnly ?? false}>
          <AutopilotAvailableProvider
            value={loaderData?.autopilotAvailable ?? false}
          >
            <SidebarProvider>
              <TooltipProvider>{children}</TooltipProvider>
            </SidebarProvider>
          </AutopilotAvailableProvider>
        </ReadOnlyProvider>
        <Toaster />
      </GlobalToastProvider>
    </ReactQueryProvider>
  );
}
