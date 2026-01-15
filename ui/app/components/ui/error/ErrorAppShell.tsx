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
