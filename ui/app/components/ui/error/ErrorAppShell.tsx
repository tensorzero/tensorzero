import { AppSidebar } from "~/components/layout/app.sidebar";
import { ContentLayout } from "~/components/layout/ContentLayout";
import {
  AppProviders,
  type AppProvidersLoaderData,
} from "~/providers/app-providers";

interface ErrorAppShellProps {
  content?: React.ReactNode;
  overlay?: React.ReactNode;
  loaderData?: AppProvidersLoaderData;
}

/**
 * Renders the app shell (sidebar + content area) for error states.
 * Used when the root loader fails but we still want to show the app frame.
 *
 * @param content - Content to render inside ContentLayout (e.g., PageNotFound)
 * @param overlay - Content to render as an overlay on top (e.g., ErrorDialog)
 * @param loaderData - Loader data to pass to providers for context (config, read-only, etc.)
 */
export function ErrorAppShell({
  content,
  overlay,
  loaderData,
}: ErrorAppShellProps) {
  return (
    <AppProviders loaderData={loaderData}>
      <div className="fixed inset-0 flex">
        <AppSidebar />
        <ContentLayout>{content}</ContentLayout>
      </div>
      {overlay}
    </AppProviders>
  );
}
