import { AppSidebar } from "~/components/layout/app.sidebar";
import { ContentLayout } from "~/components/layout/ContentLayout";
import { AppProviders } from "~/providers/app-providers";

interface RootErrorBoundaryLayoutProps {
  children?: React.ReactNode;
}

/**
 * Renders the app shell even when the root loader fails.
 * Shows sidebar + layout so the app feels present but blocked by the error,
 * rather than completely broken. Children render as an overlay on top.
 */
export function RootErrorBoundaryLayout({
  children,
}: RootErrorBoundaryLayoutProps) {
  return (
    <AppProviders>
      <div className="fixed inset-0 flex">
        <AppSidebar />
        <ContentLayout />
      </div>
      {children}
    </AppProviders>
  );
}
