import { ReactQueryProvider } from "./react-query";
import { GlobalToastProvider } from "./global-toast-provider";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { Toaster } from "~/components/ui/toaster";
import { ReadOnlyProvider } from "~/context/read-only";
import { AutopilotAvailableProvider } from "~/context/autopilot-available";

export interface AppProvidersLoaderData {
  isReadOnly?: boolean;
  autopilotAvailable?: boolean;
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
