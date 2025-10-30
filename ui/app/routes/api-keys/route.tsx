import type { Route } from "./+types/route";
import {
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { useState } from "react";
import { logger } from "~/utils/logger";
import { getPostgresClient } from "~/utils/postgres.server";
import AuthTable from "./AuthTable";
import { AuthActions } from "./AuthActions";
import { GenerateApiKeyModal } from "./GenerateApiKeyModal";
import type { KeyInfo } from "tensorzero-node";

export const handle: RouteHandle = {
  crumb: () => ["TensorZero API Keys"],
};

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "100");

  const postgresClient = await getPostgresClient();
  const apiKeysJson = await postgresClient.listApiKeys();
  const allApiKeys: KeyInfo[] = JSON.parse(apiKeysJson);

  // Paginate the results
  const apiKeys = allApiKeys.slice(offset, offset + pageSize);
  const totalApiKeys = allApiKeys.length;

  return {
    totalApiKeys,
    apiKeys,
    offset,
    pageSize,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const actionType = formData.get("action");

  if (actionType === "generate") {
    try {
      const description = formData.get("description");
      const descriptionStr =
        description && typeof description === "string" && description.trim()
          ? description.trim()
          : null;

      const postgresClient = await getPostgresClient();
      const apiKey = await postgresClient.createApiKey(descriptionStr);

      return {
        apiKey,
      };
    } catch (error) {
      logger.error("Failed to generate API key", error);
      return {
        error: "Failed to generate API key. Please try again.",
      };
    }
  }

  return {
    error: "Invalid action",
  };
}

export default function AuthPage({ loaderData }: Route.ComponentProps) {
  const navigate = useNavigate();
  const { totalApiKeys, apiKeys, offset, pageSize } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const [generateModalIsOpen, setGenerateModalIsOpen] = useState(false);

  return (
    <PageLayout>
      <PageHeader heading="TensorZero API Keys" count={totalApiKeys} />
      <SectionLayout>
        <AuthActions onGenerateKey={() => setGenerateModalIsOpen(true)} />
        <AuthTable apiKeys={apiKeys} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + pageSize >= totalApiKeys}
        />
      </SectionLayout>
      <GenerateApiKeyModal
        isOpen={generateModalIsOpen}
        onClose={() => setGenerateModalIsOpen(false)}
      />
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
