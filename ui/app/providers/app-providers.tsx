import { ReactQueryProvider } from "./react-query";
import { GlobalToastProvider } from "./global-toast-provider";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { Toaster } from "~/components/ui/toaster";

interface AppProvidersProps {
  children: React.ReactNode;
}

/**
 * Shared provider stack for the app shell.
 * Used by both the main App and ErrorAppShell to ensure consistency.
 */
export function AppProviders({ children }: AppProvidersProps) {
  return (
    <ReactQueryProvider>
      <GlobalToastProvider>
        <SidebarProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </SidebarProvider>
        <Toaster />
      </GlobalToastProvider>
    </ReactQueryProvider>
  );
}
