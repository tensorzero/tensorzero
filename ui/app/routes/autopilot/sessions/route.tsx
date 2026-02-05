import { Plus } from "lucide-react";
import { Suspense } from "react";
import type { Route } from "./+types/route";
import { Await, data, useLocation, useNavigate } from "react-router";
import { useTensorZeroStatusFetcher } from "~/routes/api/tensorzero/status";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { ActionBar } from "~/components/layout/ActionBar";
import { Button } from "~/components/ui/button";
import PageButtons from "~/components/utils/PageButtons";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
import { SessionsTableRows } from "../AutopilotSessionsTable";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import type { Session } from "~/types/tensorzero";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableAsyncErrorState,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

const MAX_PAGE_SIZE = 50;
const DEFAULT_PAGE_SIZE = 20;

export type SessionsData = {
  sessions: Session[];
  hasMore: boolean;
};

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

  // Return promise WITHOUT awaiting - enables streaming/skeleton loading
  const sessionsDataPromise = client
    .listAutopilotSessions({
      limit: limit + 1,
      offset,
    })
    .then((response) => {
      const hasMore = response.sessions.length > limit;
      const sessions = response.sessions.slice(0, limit);
      return { sessions, hasMore };
    });

  return {
    sessionsData: sessionsDataPromise,
    offset,
    limit,
  };
}

// Skeleton rows for loading state - matches table columns (Session ID, Summary, Created)
function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-5 w-24" />
          </TableCell>
          <TableCell className="max-w-xs">
            <Skeleton className="h-5 w-48" />
          </TableCell>
          <TableCell className="w-52 whitespace-nowrap">
            <Skeleton className="h-5 w-36" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

export default function AutopilotSessionsPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { status } = useTensorZeroStatusFetcher();
  const { sessionsData, offset, limit } = loaderData;
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
        <ActionBar>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/autopilot/sessions/new")}
          >
            <Plus className="text-fg-tertiary mr-2 h-4 w-4" />
            New Session
          </Button>
        </ActionBar>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-36">Session ID</TableHead>
              <TableHead>Summary</TableHead>
              <TableHead className="w-52 whitespace-nowrap">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <Suspense key={location.search} fallback={<SkeletonRows />}>
              <Await
                resolve={sessionsData}
                errorElement={
                  <TableAsyncErrorState
                    colSpan={3}
                    defaultMessage="Failed to load sessions"
                  />
                }
              >
                {({ sessions }) => (
                  <SessionsTableRows
                    sessions={sessions}
                    gatewayVersion={gatewayVersion}
                    uiVersion={uiVersion}
                  />
                )}
              </Await>
            </Suspense>
          </TableBody>
        </Table>
        <Suspense key={location.search} fallback={<PageButtons disabled />}>
          <Await resolve={sessionsData} errorElement={<PageButtons disabled />}>
            {({ hasMore }) => (
              <PageButtons
                onPreviousPage={handlePreviousPage}
                onNextPage={handleNextPage}
                disablePrevious={offset <= 0}
                disableNext={!hasMore}
              />
            )}
          </Await>
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
