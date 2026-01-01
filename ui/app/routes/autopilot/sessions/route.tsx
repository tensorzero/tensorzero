import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import { useTensorZeroStatusFetcher } from "~/routes/api/tensorzero/status";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import { logger } from "~/utils/logger";
import AutopilotSessionsTable from "../AutopilotSessionsTable";
import { getAutopilotClient } from "~/utils/tensorzero.server";

const MAX_PAGE_SIZE = 50;
const DEFAULT_PAGE_SIZE = 20;

function parseInteger(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const limitParam = parseInteger(
    url.searchParams.get("limit"),
    DEFAULT_PAGE_SIZE,
  );
  const offsetParam = parseInteger(url.searchParams.get("offset"), 0);
  const limit = Math.max(1, limitParam);
  const offset = Math.max(0, offsetParam);

  if (limit > MAX_PAGE_SIZE) {
    throw data(`Limit cannot exceed ${MAX_PAGE_SIZE}`, { status: 400 });
  }

  const client = getAutopilotClient();
  // Fetch limit + 1 to detect if more pages exist
  const response = await client.listAutopilotSessions({
    limit: limit + 1,
    offset,
  });

  const hasMore = response.sessions.length > limit;
  const sessions = response.sessions.slice(0, limit);

  return {
    sessions,
    offset,
    limit,
    hasMore,
  };
}

export default function AutopilotSessionsPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const { status } = useTensorZeroStatusFetcher();
  const { sessions, offset, limit, hasMore } = loaderData;
  const gatewayVersion = status?.version;
  const uiVersion = __APP_VERSION__;

  const updateOffset = (nextOffset: number) => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(nextOffset));
    searchParams.set("limit", String(limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleNextPage = () => {
    updateOffset(offset + limit);
  };

  const handlePreviousPage = () => {
    updateOffset(Math.max(0, offset - limit));
  };

  return (
    <PageLayout>
      <PageHeader heading="Autopilot Sessions" />
      <SectionLayout>
        <AutopilotSessionsTable
          sessions={sessions}
          gatewayVersion={gatewayVersion}
          uiVersion={uiVersion}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={!hasMore}
        />
      </SectionLayout>
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
