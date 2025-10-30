import type { Route } from "./+types/route";
import {
  data,
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

export const handle: RouteHandle = {
  crumb: () => ["TensorZero API Keys"],
};

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "100");

  if (pageSize > 10000) {
    throw data("Page size cannot exceed 10,000", { status: 400 });
  }

  const postgresClient = await getPostgresClient();
  const apiKeys = await postgresClient.listApiKeys(pageSize, offset);

  return {
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

  if (actionType === "delete") {
    try {
      const publicId = formData.get("publicId");
      if (typeof publicId !== "string") {
        return {
          error: "Public ID is required",
        };
      }

      const postgresClient = await getPostgresClient();
      await postgresClient.disableApiKey(publicId);

      return {
        success: true,
      };
    } catch (error) {
      logger.error("Failed to disable API key", error);
      return {
        error: "Failed to disable API key. Please try again.",
      };
    }
  }

  return {
    error: "Invalid action",
  };
}

export default function AuthPage({ loaderData }: Route.ComponentProps) {
  const navigate = useNavigate();
  const { apiKeys, offset, pageSize } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(Math.max(0, offset - pageSize)));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const [generateModalIsOpen, setGenerateModalIsOpen] = useState(false);
  const [modalKey, setModalKey] = useState(0);

  const handleOpenModal = () => {
    setModalKey((prev) => prev + 1);
    setGenerateModalIsOpen(true);
  };

  const handleCloseModal = () => {
    setGenerateModalIsOpen(false);
    // Reset to first page after creating API key
    if (offset > 0) {
      const searchParams = new URLSearchParams(window.location.search);
      searchParams.set("offset", "0");
      navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
    }
  };

  return (
    <PageLayout>
      <PageHeader heading="TensorZero API Keys" />
      <SectionLayout>
        <AuthActions onGenerateKey={handleOpenModal} />
        <AuthTable apiKeys={apiKeys} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={apiKeys.length < pageSize}
        />
      </SectionLayout>
      <GenerateApiKeyModal
        key={modalKey}
        isOpen={generateModalIsOpen}
        onClose={handleCloseModal}
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
