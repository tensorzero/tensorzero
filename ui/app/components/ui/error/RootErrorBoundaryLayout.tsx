import { AppSidebar } from "~/components/layout/app.sidebar";
import { ContentLayout } from "~/components/layout/ContentLayout";
import { SidebarProvider } from "~/components/ui/sidebar";
import { TooltipProvider } from "~/components/ui/tooltip";
import { ReactQueryProvider } from "~/providers/react-query";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
import { Toaster } from "~/components/ui/toaster";

interface RootErrorBoundaryLayoutProps {
  children?: React.ReactNode;
}

// Provides necessary context providers without requiring config
export function RootErrorBoundaryLayout({
  children,
}: RootErrorBoundaryLayoutProps) {
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
