import type { Route } from "./+types/route";
import { Suspense, useState } from "react";
import {
  Await,
  data,
  useAsyncError,
  useLocation,
  useNavigate,
  type RouteHandle,
} from "react-router";
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
} from "~/utils/postgres.server";
import AuthTable from "./AuthTable";
import { AuthActions } from "./AuthActions";
import { GenerateApiKeyModal } from "./GenerateApiKeyModal";
import { PostgresRequiredState } from "~/components/ui/PostgresRequiredState";
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

export type ApiKeysData = {
  apiKeys: KeyInfo[];
  offset: number;
  limit: number;
};

function ApiKeysPageHeader() {
  return <PageHeader heading="TensorZero API Keys" />;
}

function ApiKeysContentSkeleton() {
  return (
    <>
      <ApiKeysPageHeader />
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
    </>
  );
}

function ApiKeysErrorState() {
  const error = useAsyncError();
  return (
    <>
      <ApiKeysPageHeader />
      <SectionLayout>
        <PageErrorContent error={error} />
      </SectionLayout>
    </>
  );
}

async function fetchApiKeys(
  limit: number,
  offset: number,
): Promise<ApiKeysData> {
  const postgresClient = await getPostgresClient();
  const apiKeys = await postgresClient.listApiKeys(limit, offset);
  return { apiKeys, offset, limit };
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const limit = parseInt(searchParams.get("limit") || "100");

  if (limit > 10000) {
    throw data("Limit cannot exceed 10,000", { status: 400 });
  }

  if (!isPostgresAvailable()) {
    return {
      postgresAvailable: false as const,
      apiKeysData: null,
    };
  }

  return {
    postgresAvailable: true as const,
    apiKeysData: fetchApiKeys(limit, offset),
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
      return {
        error: "Failed to update API key description. Please try again.",
      };
    }
  }

  return {
    error: "Invalid action",
  };
}

function ApiKeysContent({
  data,
  onOpenModal,
}: {
  data: ApiKeysData;
  onOpenModal: () => void;
}) {
  const { apiKeys, offset, limit } = data;
  const navigate = useNavigate();

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

  return (
    <>
      <ApiKeysPageHeader />
      <SectionLayout>
        <AuthActions onGenerateKey={onOpenModal} />
        <AuthTable apiKeys={apiKeys} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={apiKeys.length < limit}
        />
      </SectionLayout>
    </>
  );
}

export default function ApiKeysPage({ loaderData }: Route.ComponentProps) {
  const { postgresAvailable, apiKeysData } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();
  const [generateModalIsOpen, setGenerateModalIsOpen] = useState(false);
  const [modalKey, setModalKey] = useState(0);

  if (!postgresAvailable) {
    return <PostgresRequiredState />;
  }

  const handleOpenModal = () => {
    setModalKey((prev) => prev + 1);
    setGenerateModalIsOpen(true);
  };

  const handleCloseModal = () => {
    setGenerateModalIsOpen(false);
    // Reset to first page after creating API key
    const searchParams = new URLSearchParams(window.location.search);
    const offset = parseInt(searchParams.get("offset") || "0");
    if (offset > 0) {
      searchParams.set("offset", "0");
      navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
    }
  };

  return (
    <PageLayout>
      <Suspense key={location.key} fallback={<ApiKeysContentSkeleton />}>
        <Await resolve={apiKeysData} errorElement={<ApiKeysErrorState />}>
          {(resolvedData) => (
            <ApiKeysContent data={resolvedData} onOpenModal={handleOpenModal} />
          )}
        </Await>
      </Suspense>
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
  return <PageErrorContent error={error} />;
}
