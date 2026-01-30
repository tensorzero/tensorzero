import { AlertTriangle } from "lucide-react";
import type { Session } from "~/types/tensorzero";
import {
  Table,
  TableBody,
  TableCell,
  TableEmptyState,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";

type AutopilotSessionsTableProps = {
  sessions: Session[];
  gatewayVersion?: string;
  uiVersion?: string;
};

function shouldWarnVersionMismatch({
  sessionVersion,
  gatewayVersion,
  uiVersion,
}: {
  sessionVersion: string;
  gatewayVersion?: string;
  uiVersion?: string;
}) {
  if (!gatewayVersion || !uiVersion) {
    return false;
  }

  return sessionVersion !== gatewayVersion || sessionVersion !== uiVersion;
}

function buildMismatchMessage({
  sessionVersion,
  gatewayVersion,
  uiVersion,
}: {
  sessionVersion: string;
  gatewayVersion: string;
  uiVersion: string;
}) {
  return (
    <div className="flex flex-col items-start gap-1">
      <span className="flex items-center gap-1">
        <AlertTriangle className="h-4 w-4" />
        Version Mismatch
      </span>
      <span className="text-fg-muted">
        The agent's performance might not be optimal.
      </span>
      <hr className="bg-fg-muted my-1 h-px w-full border-0" />
      <table className="inline-table w-auto self-start text-xs">
        <tbody>
          <tr>
            <td className="text-fg-muted pr-2 align-top">Autopilot Session</td>
            <td className="font-mono text-xs whitespace-nowrap">
              {sessionVersion}
            </td>
          </tr>
          <tr>
            <td className="text-fg-muted pr-2 align-top">TensorZero Gateway</td>
            <td className="font-mono text-xs whitespace-nowrap">
              {gatewayVersion}
            </td>
          </tr>
          <tr>
            <td className="text-fg-muted pr-2 align-top">TensorZero UI</td>
            <td className="font-mono text-xs whitespace-nowrap">{uiVersion}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function SessionIdCell({
  session,
  gatewayVersion,
  uiVersion,
}: {
  session: Session;
  gatewayVersion?: string;
  uiVersion?: string;
}) {
  const showWarning = shouldWarnVersionMismatch({
    sessionVersion: session.tensorzero_version,
    gatewayVersion,
    uiVersion,
  });

  const sessionUrl = `/autopilot/sessions/${session.id}`;

  return (
    <div className="flex items-center gap-2">
      <TableItemShortUuid id={session.id} link={sessionUrl} />
      {showWarning && gatewayVersion && uiVersion && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="inline-flex cursor-help items-center text-yellow-600"
              aria-label="Version mismatch"
            >
              <AlertTriangle className="h-4 w-4" />
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            {buildMismatchMessage({
              sessionVersion: session.tensorzero_version,
              gatewayVersion,
              uiVersion,
            })}
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

// Renders just the table rows (for use with Suspense inside TableBody)
export function SessionsTableRows({
  sessions,
  gatewayVersion,
  uiVersion,
}: AutopilotSessionsTableProps) {
  if (sessions.length === 0) {
    return <TableEmptyState message="No sessions found" />;
  }

  return (
    <>
      {sessions.map((session) => (
        <TableRow key={session.id}>
          <TableCell>
            <SessionIdCell
              session={session}
              gatewayVersion={gatewayVersion}
              uiVersion={uiVersion}
            />
          </TableCell>
          <TableCell className="w-0 text-right whitespace-nowrap">
            <TableItemTime timestamp={session.created_at} />
          </TableCell>
          <TableCell className="w-0 text-right whitespace-nowrap">
            {session.last_event_at ? (
              <TableItemTime timestamp={session.last_event_at} />
            ) : (
              <span className="text-fg-muted">—</span>
            )}
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

export default function AutopilotSessionsTable({
  sessions,
  gatewayVersion,
  uiVersion,
}: AutopilotSessionsTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Session ID</TableHead>
          <TableHead className="w-0 text-right whitespace-nowrap">
            Created
          </TableHead>
          <TableHead className="w-0 text-right whitespace-nowrap">
            Last Activity
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <SessionsTableRows
          sessions={sessions}
          gatewayVersion={gatewayVersion}
          uiVersion={uiVersion}
        />
      </TableBody>
    </Table>
  );
}
