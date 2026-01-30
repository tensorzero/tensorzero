import { ChevronDown, ChevronUp, Plus } from "lucide-react";
import { Suspense, use } from "react";
import type { Route } from "./+types/route";
import { data, useLocation, useNavigate } from "react-router";
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
import type { Session, SessionSortField, SortOrder } from "~/types/tensorzero";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

const MAX_PAGE_SIZE = 50;
const DEFAULT_PAGE_SIZE = 20;
const DEFAULT_SORT_BY: SessionSortField = "last_event_at";
const DEFAULT_SORT_ORDER: SortOrder = "desc";

export type SessionsData = {
  sessions: Session[];
  hasMore: boolean;
};

function parseInteger(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function parseSortBy(value: string | null): SessionSortField {
  if (value === "created_at" || value === "last_event_at") {
    return value;
  }
  return DEFAULT_SORT_BY;
}

function parseSortOrder(value: string | null): SortOrder {
  if (value === "asc" || value === "desc") {
    return value;
  }
  return DEFAULT_SORT_ORDER;
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
  const sortBy = parseSortBy(url.searchParams.get("sort_by"));
  const sortOrder = parseSortOrder(url.searchParams.get("sort_order"));

  if (limit > MAX_PAGE_SIZE) {
    throw data(`Limit cannot exceed ${MAX_PAGE_SIZE}`, { status: 400 });
  }

  const client = getAutopilotClient();

  // Return promise WITHOUT awaiting - enables streaming/skeleton loading
  const sessionsDataPromise = client
    .listAutopilotSessions({
      limit: limit + 1,
      offset,
      sort_by: sortBy,
      sort_order: sortOrder,
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
    sortBy,
    sortOrder,
  };
}

// Skeleton rows for loading state - matches table columns (Session ID, Created, Last Activity)
function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-5 w-24" />
          </TableCell>
          <TableCell className="w-0 text-right whitespace-nowrap">
            <Skeleton className="ml-auto h-5 w-36" />
          </TableCell>
          <TableCell className="w-0 text-right whitespace-nowrap">
            <Skeleton className="ml-auto h-5 w-36" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

// Resolves promise and renders table rows
function TableBodyContent({
  data,
  gatewayVersion,
  uiVersion,
}: {
  data: Promise<SessionsData>;
  gatewayVersion?: string;
  uiVersion?: string;
}) {
  const { sessions } = use(data);

  return (
    <SessionsTableRows
      sessions={sessions}
      gatewayVersion={gatewayVersion}
      uiVersion={uiVersion}
    />
  );
}

// Resolves promise and renders pagination
function PaginationContent({
  data,
  offset,
  onPreviousPage,
  onNextPage,
}: {
  data: Promise<SessionsData>;
  offset: number;
  onPreviousPage: () => void;
  onNextPage: () => void;
}) {
  const { hasMore } = use(data);

  return (
    <PageButtons
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
      disablePrevious={offset <= 0}
      disableNext={!hasMore}
    />
  );
}

// Sortable column header component
function SortableHeader({
  label,
  field,
  currentSortBy,
  currentSortOrder,
  onSort,
  className,
}: {
  label: string;
  field: SessionSortField;
  currentSortBy: SessionSortField;
  currentSortOrder: SortOrder;
  onSort: (field: SessionSortField) => void;
  className?: string;
}) {
  const isActive = currentSortBy === field;

  return (
    <TableHead
      className={`cursor-pointer select-none ${className ?? ""}`}
      onClick={() => onSort(field)}
    >
      <div className="flex items-center justify-end gap-1">
        {label}
        {isActive ? (
          currentSortOrder === "asc" ? (
            <ChevronUp className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )
        ) : (
          <div className="flex flex-col">
            <ChevronUp className="h-2 w-2 opacity-40" />
            <ChevronDown className="h-2 w-2 opacity-40" />
          </div>
        )}
      </div>
    </TableHead>
  );
}

export default function AutopilotSessionsPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { status } = useTensorZeroStatusFetcher();
  const { sessionsData, offset, limit, sortBy, sortOrder } = loaderData;
  const gatewayVersion = status?.version;
  const uiVersion = __APP_VERSION__;

  const updateSearchParams = (updates: Record<string, string>) => {
    const searchParams = new URLSearchParams(window.location.search);
    for (const [key, value] of Object.entries(updates)) {
      searchParams.set(key, value);
    }
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleSort = (field: SessionSortField) => {
    // If clicking the same field, toggle order. Otherwise, sort desc by the new field.
    const newOrder =
      sortBy === field ? (sortOrder === "desc" ? "asc" : "desc") : "desc";
    updateSearchParams({
      sort_by: field,
      sort_order: newOrder,
      offset: "0", // Reset to first page when changing sort
    });
  };

  const handleNextPage = () => {
    updateSearchParams({
      offset: String(offset + limit),
      limit: String(limit),
    });
  };

  const handlePreviousPage = () => {
    updateSearchParams({
      offset: String(Math.max(0, offset - limit)),
      limit: String(limit),
    });
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
              <TableHead>Session ID</TableHead>
              <SortableHeader
                label="Created"
                field="created_at"
                currentSortBy={sortBy}
                currentSortOrder={sortOrder}
                onSort={handleSort}
                className="w-0 text-right whitespace-nowrap"
              />
              <SortableHeader
                label="Last Activity"
                field="last_event_at"
                currentSortBy={sortBy}
                currentSortOrder={sortOrder}
                onSort={handleSort}
                className="w-0 text-right whitespace-nowrap"
              />
            </TableRow>
          </TableHeader>
          <TableBody>
            <Suspense key={location.search} fallback={<SkeletonRows />}>
              <TableBodyContent
                data={sessionsData}
                gatewayVersion={gatewayVersion}
                uiVersion={uiVersion}
              />
            </Suspense>
          </TableBody>
        </Table>
        <Suspense key={location.search} fallback={<PageButtons disabled />}>
          <PaginationContent
            data={sessionsData}
            offset={offset}
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
          />
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
