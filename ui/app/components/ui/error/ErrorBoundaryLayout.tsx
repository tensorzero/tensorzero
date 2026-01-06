import { AppSidebar } from "~/components/layout/app.sidebar";
import { ContentLayout } from "~/components/layout/ContentLayout";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { ReactQueryProvider } from "~/providers/react-query";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
import { Toaster } from "~/components/ui/toaster";

interface ErrorBoundaryLayoutProps {
  children?: React.ReactNode;
}

/**
 * Shared layout for error boundary states that need the sidebar visible.
 * Provides all necessary context providers and renders the sidebar.
 *
 * Usage:
 * ```tsx
 * <ErrorBoundaryLayout>
 *   <ErrorDialog ...>
 *     <YourErrorContent />
 *   </ErrorDialog>
 * </ErrorBoundaryLayout>
 * ```
 */
export function ErrorBoundaryLayout({ children }: ErrorBoundaryLayoutProps) {
  return (
    <ReactQueryProvider>
      <GlobalToastProvider>
        <SidebarProvider>
          <TooltipProvider>
            <div className="fixed inset-0 flex">
              <AppSidebar />
              <ContentLayout />
            </div>
            {children}
          </TooltipProvider>
        </SidebarProvider>
        <Toaster />
      </GlobalToastProvider>
    </ReactQueryProvider>
  );
}
