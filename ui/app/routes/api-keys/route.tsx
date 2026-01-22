import type { Route } from "./+types/route";
import { Await, data, useNavigate, type RouteHandle } from "react-router";
import { Suspense, useState } from "react";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { PageErrorContent } from "~/components/ui/error";
import {
  getPostgresClient,
  isPostgresAvailable,
  PostgresConnectionError,
} from "~/utils/postgres.server";
import AuthTable from "./AuthTable";
import { AuthActions } from "./AuthActions";
import { GenerateApiKeyModal } from "./GenerateApiKeyModal";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorInlineCode,
  TroubleshootingSection,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertTriangle, Database } from "lucide-react";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { KeyInfo } from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: () => ["TensorZero API Keys"],
};

type ApiKeysResult =
  | { status: "success"; apiKeys: KeyInfo[] }
  | { status: "connection_error"; message: string };

async function fetchApiKeys(
  limit: number,
  offset: number,
): Promise<ApiKeysResult> {
  try {
    const postgresClient = await getPostgresClient();
    const apiKeys = await postgresClient.listApiKeys(limit, offset);
    return { status: "success", apiKeys };
  } catch (error) {
    if (error instanceof PostgresConnectionError) {
      logger.error("Failed to connect to Postgres", error);
      return { status: "connection_error", message: error.message };
    }
    throw error;
  }
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "100");

  if (limit > 10000) {
    throw data("Limit cannot exceed 10,000", { status: 400 });
  }

  // Quick sync check - if not configured, return immediately
  if (!isPostgresAvailable()) {
    return {
      postgresAvailable: false as const,
      apiKeysPromise: null,
      offset: 0,
      limit: 0,
    };
  }

  // Return immediately, defer the actual postgres work
  return {
    postgresAvailable: true as const,
    apiKeysPromise: fetchApiKeys(limit, offset),
    offset,
    limit,
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
      if (error instanceof PostgresConnectionError) {
        return {
          error: "Unable to connect to database. Please try again later.",
        };
      }
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
      if (error instanceof PostgresConnectionError) {
        return {
          error: "Unable to connect to database. Please try again later.",
        };
      }
      return {
        error: "Failed to disable API key. Please try again.",
      };
    }
  }

  if (actionType === "update") {
    try {
      const publicId = formData.get("publicId");
      if (typeof publicId !== "string" || !publicId.trim()) {
        return {
          error: "Public ID is required",
        };
      }

      const description = formData.get("description");
      const descriptionStr =
        description && typeof description === "string"
          ? description.trim() || null
          : null;

      const postgresClient = await getPostgresClient();
      await postgresClient.updateApiKeyDescription(
        publicId.trim(),
        descriptionStr,
      );

      return {
        success: true,
      };
    } catch (error) {
      logger.error("Failed to update API key description", error);
      if (error instanceof PostgresConnectionError) {
        return {
          error: "Unable to connect to database. Please try again later.",
        };
      }
      return {
        error: "Failed to update API key description. Please try again.",
      };
    }
  }

  return {
    error: "Invalid action",
  };
}

function DisabledPageContent() {
  return (
    <>
      <AuthActions disabled />
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-0 whitespace-nowrap">Public ID</TableHead>
            <TableHead>Description</TableHead>
            <TableHead className="w-0 whitespace-nowrap">Created</TableHead>
            <TableHead className="w-0"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {[1, 2, 3].map((i) => (
            <TableRow key={i}>
              <TableCell>
                <div className="bg-muted h-4 w-24 rounded" />
              </TableCell>
              <TableCell>
                <div className="bg-muted h-4 w-48 rounded" />
              </TableCell>
              <TableCell>
                <div className="bg-muted h-4 w-20 rounded" />
              </TableCell>
              <TableCell>
                <div className="flex gap-2">
                  <div className="bg-muted h-8 w-8 rounded" />
                  <div className="bg-muted h-8 w-8 rounded" />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <PageButtons
        onPreviousPage={() => {}}
        onNextPage={() => {}}
        disablePrevious
        disableNext
      />
    </>
  );
}

function ApiKeysLoadingState() {
  return (
    <PageLayout>
      <PageHeader heading="TensorZero API Keys" />
      <SectionLayout>
        <div className="flex flex-wrap gap-2">
          <Skeleton className="h-8 w-32" />
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-0 whitespace-nowrap">Public ID</TableHead>
              <TableHead>Description</TableHead>
              <TableHead className="w-0 whitespace-nowrap">Created</TableHead>
              <TableHead className="w-0"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3].map((i) => (
              <TableRow key={i}>
                <TableCell>
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-48" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-20" />
                </TableCell>
                <TableCell>
                  <div className="flex gap-2">
                    <Skeleton className="h-8 w-8" />
                    <Skeleton className="h-8 w-8" />
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="mt-4 flex items-center justify-center gap-2">
          <Skeleton className="h-9 w-9 rounded-md" />
          <Skeleton className="h-9 w-9 rounded-md" />
        </div>
      </SectionLayout>
    </PageLayout>
  );
}

function PostgresNotConfiguredState() {
  return (
    <PageLayout className="relative min-h-[calc(100vh-4rem)]">
      <PageHeader heading="TensorZero API Keys" />
      <SectionLayout className="opacity-50">
        <DisabledPageContent />
      </SectionLayout>
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-8 pt-16 pb-20">
        <div className="pointer-events-auto">
          <ErrorContentCard>
            <ErrorContentHeader
              icon={Database}
              title="Postgres Not Configured"
              description="Postgres database connection is required to manage API keys. Set the TENSORZERO_POSTGRES_URL environment variable and restart the UI server."
            />
          </ErrorContentCard>
        </div>
      </div>
    </PageLayout>
  );
}

function PostgresConnectionErrorState({ message }: { message: string }) {
  return (
    <PageLayout className="relative min-h-[calc(100vh-4rem)]">
      <PageHeader heading="TensorZero API Keys" />
      <SectionLayout className="opacity-50">
        <DisabledPageContent />
      </SectionLayout>
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-8 pt-16 pb-20">
        <div className="pointer-events-auto">
          <ErrorContentCard>
            <ErrorContentHeader
              icon={AlertTriangle}
              title="Unable to Connect to Postgres"
              description={message}
            />
            <TroubleshootingSection>
              <span>Verify Postgres is running and accessible</span>
              <span>
                Check that{" "}
                <ErrorInlineCode>TENSORZERO_POSTGRES_URL</ErrorInlineCode> is
                correct
              </span>
              <span>Ensure network connectivity to the database host</span>
              <span>Verify database credentials and permissions</span>
            </TroubleshootingSection>
          </ErrorContentCard>
        </div>
      </div>
    </PageLayout>
  );
}

function ApiKeysContent({
  result,
  offset,
  limit,
}: {
  result: ApiKeysResult;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [generateModalIsOpen, setGenerateModalIsOpen] = useState(false);
  const [modalKey, setModalKey] = useState(0);

  if (result.status === "connection_error") {
    return <PostgresConnectionErrorState message={result.message} />;
  }

  const { apiKeys } = result;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(Math.max(0, offset - limit)));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleOpenModal = () => {
    setModalKey((prev) => prev + 1);
    setGenerateModalIsOpen(true);
  };

  const handleCloseModal = () => {
    setGenerateModalIsOpen(false);
    if (offset > 0) {
      const searchParams = new URLSearchParams(window.location.search);
      searchParams.set("offset", "0");
      searchParams.set("limit", String(limit));
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
          disableNext={apiKeys.length < limit}
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

export default function AuthPage({ loaderData }: Route.ComponentProps) {
  const { postgresAvailable, apiKeysPromise, offset, limit } = loaderData;

  if (!postgresAvailable) {
    return <PostgresNotConfiguredState />;
  }

  return (
    <Suspense fallback={<ApiKeysLoadingState />}>
      <Await resolve={apiKeysPromise}>
        {(result) => (
          <ApiKeysContent result={result} offset={offset} limit={limit} />
        )}
      </Await>
    </Suspense>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <PageErrorContent error={error} />;
}
