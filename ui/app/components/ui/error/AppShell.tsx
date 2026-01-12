import { AppSidebar } from "~/components/layout/app.sidebar";
import { ContentLayout } from "~/components/layout/ContentLayout";
import { AppProviders } from "~/providers/app-providers";

interface AppShellProps {
  content?: React.ReactNode;
  overlay?: React.ReactNode;
}

/**
 * Renders the app shell (sidebar + content area) for error states.
 * Used when the root loader fails but we still want to show the app frame.
 *
 * @param content - Content to render inside ContentLayout (e.g., PageNotFound)
 * @param overlay - Content to render as an overlay on top (e.g., ErrorDialog)
 */
export function AppShell({ content, overlay }: AppShellProps) {
  return (
    <AppProviders>
      <div className="fixed inset-0 flex">
        <AppSidebar />
        <ContentLayout>{content}</ContentLayout>
      </div>
      {overlay}
    </AppProviders>
  );
}
