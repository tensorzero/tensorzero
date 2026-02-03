import { ReactQueryProvider } from "./react-query";
import { GlobalToastProvider } from "./global-toast-provider";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { Toaster } from "~/components/ui/toaster";
import { ReadOnlyProvider } from "~/context/read-only";
import { AutopilotAvailableProvider } from "~/context/autopilot-available";
import { ConfigProvider, EMPTY_CONFIG } from "~/context/config";
import type { UiConfig } from "~/types/tensorzero";

export interface AppProvidersLoaderData {
  isReadOnly?: boolean;
  autopilotAvailable?: boolean;
  config?: UiConfig;
}

interface AppProvidersProps {
  children: React.ReactNode;
  loaderData?: AppProvidersLoaderData;
}

/**
 * Shared provider stack for the app shell.
 * Used by both the main App and ErrorAppShell to ensure consistency.
 */
export function AppProviders({ children, loaderData }: AppProvidersProps) {
  return (
    <ReactQueryProvider>
      <GlobalToastProvider>
        <ReadOnlyProvider value={loaderData?.isReadOnly ?? false}>
          <AutopilotAvailableProvider
            value={loaderData?.autopilotAvailable ?? false}
          >
            <ConfigProvider value={loaderData?.config ?? EMPTY_CONFIG}>
              <SidebarProvider>
                <TooltipProvider delayDuration={250}>
                  {children}
                </TooltipProvider>
              </SidebarProvider>
            </ConfigProvider>
          </AutopilotAvailableProvider>
        </ReadOnlyProvider>
        <Toaster />
      </GlobalToastProvider>
    </ReactQueryProvider>
  );
}
