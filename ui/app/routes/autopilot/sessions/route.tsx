import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
} from "react-router";
import { useTensorZeroStatusFetcher } from "~/routes/api/tensorzero/status";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import { logger } from "~/utils/logger";
import type { Session } from "~/types/tensorzero";
import AutopilotSessionsTable from "../AutopilotSessionsTable";

const MAX_PAGE_SIZE = 50;
const DEFAULT_PAGE_SIZE = 20;
const MOCK_SESSIONS = buildMockSessions(57);

const ORGANIZATION_IDS = [
  "0c4f3c9a-2e1b-4b9c-8d2a-8f7c2e7c0001",
  "1b6a7d2f-3c7a-4d1f-8e2b-9c0d1e2f0002",
  "2d5e6a7b-4c8d-5e2f-9f3c-0a1b2c3d0003",
];
const WORKSPACE_IDS = [
  "7a3b1c2d-5e6f-4a9b-8c7d-1e2f3a4b0001",
  "8c4d2e3f-6a7b-5c8d-9e0f-2a3b4c5d0002",
];
const DEPLOYMENT_IDS = [
  "3f2e1d0c-9b8a-7c6d-5e4f-3a2b1c0d0001",
  "4a3b2c1d-0e9f-8d7c-6b5a-4c3d2e1f0002",
  "5b4c3d2e-1f0a-9e8d-7c6b-5a4b3c2d0003",
];
const TENSORZERO_VERSIONS = ["2026.1.0", "2026.2.7", "2026.3.4"];

export const handle: RouteHandle = {
  crumb: () => ["Sessions"],
};

function buildUuid(prefix: string, index: number) {
  return `${prefix}-0000-0000-0000-${index.toString(16).padStart(12, "0")}`;
}

function buildMockSessions(count: number): Session[] {
  const baseTime = new Date("2024-08-15T16:30:00Z").getTime();
  const stepMs = 45 * 60 * 1000;
  return Array.from({ length: count }, (_, index) => ({
    id: buildUuid("d1a0b0c0", index + 1),
    organization_id: ORGANIZATION_IDS[index % ORGANIZATION_IDS.length],
    workspace_id: WORKSPACE_IDS[index % WORKSPACE_IDS.length],
    deployment_id: DEPLOYMENT_IDS[index % DEPLOYMENT_IDS.length],
    tensorzero_version: TENSORZERO_VERSIONS[index % TENSORZERO_VERSIONS.length],
    created_at: new Date(baseTime - index * stepMs).toISOString(),
  }));
}

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

  const sessions = MOCK_SESSIONS.slice(offset, offset + limit);

  return {
    sessions,
    totalCount: MOCK_SESSIONS.length,
    offset,
    limit,
  };
}

export default function AutopilotSessionsPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const { status } = useTensorZeroStatusFetcher();
  const { sessions, totalCount, offset, limit } = loaderData;
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
      <PageHeader heading="Autopilot Sessions" count={totalCount} />
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
          disableNext={offset + limit >= totalCount}
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
