import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, type RouteHandle } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import EventStream from "~/components/autopilot/EventStream";
import { logger } from "~/utils/logger";
import type { Event } from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.session_id!, isIdentifier: true }],
};

function buildMockEvents(sessionId: string): Event[] {
  const baseTime = new Date("2024-08-15T16:00:00Z").getTime();
  const stepMs = 2 * 60 * 1000;

  return [
    {
      id: "b8c1f0a2-1d2e-4f3a-8b4c-5d6e7f8090a1",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 0).toISOString(),
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "Draft a release plan." }],
      },
    },
    {
      id: "c9d0e1f2-2a3b-4c5d-9e6f-708192a3b4c5",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 1).toISOString(),
      payload: {
        type: "status_update",
        status_update: { type: "text", text: "Thinking through dependencies" },
      },
    },
    {
      id: "d0e1f2a3-3b4c-4d5e-8f6a-8192a3b4c5d6",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 2).toISOString(),
      payload: {
        type: "tool_call",
        id: "tool-call-1",
        name: "search_wikipedia",
        arguments: '{"query":"TensorZero"}',
      },
    },
    {
      id: "e1f2a3b4-4c5d-4e6f-9a7b-92a3b4c5d6e7",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 3).toISOString(),
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "0194e8b2-7c6a-7b9d-8c7f-1f2a3b4c5d6e",
      },
    },
    {
      id: "f2a3b4c5-5d6e-4f7a-8b9c-a3b4c5d6e7f8",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 4).toISOString(),
      payload: {
        type: "tool_result",
        tool_call_event_id: "0194e8b2-7c6a-7b9d-8c7f-1f2a3b4c5d6e",
        outcome: {
          type: "success",
          id: "tool-result-1",
          name: "search_wikipedia",
          result: "Found background details for TensorZero.",
        },
      },
    },
    {
      id: "0a1b2c3d-6e7f-4a8b-9c0d-b4c5d6e7f809",
      session_id: sessionId,
      created_at: new Date(baseTime + stepMs * 5).toISOString(),
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Here is a concise release plan with milestones and risks.",
          },
        ],
      },
    },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    throw data("Session ID is required", { status: 400 });
  }

  const events = buildMockEvents(sessionId).sort(
    (a, b) =>
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );

  return {
    sessionId,
    events,
  };
}

export default function AutopilotSessionEventsPage({
  loaderData,
}: Route.ComponentProps) {
  const { sessionId, events } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Autopilot Session" name={sessionId} />
      <SectionLayout>
        <EventStream events={events} />
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
