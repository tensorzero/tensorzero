import type { Meta, StoryObj } from "@storybook/react-vite";
import EventStream from "./EventStream";
import type { GatewayEvent, JsonValue } from "~/types/tensorzero";
import { AutopilotSessionProvider } from "~/contexts/AutopilotSessionContext";

const baseTime = new Date("2026-04-12T10:00:00Z").getTime();
const sessionId = "d1a0b0c0-0000-0000-0000-000000000001";

function at(index: number): string {
  return new Date(baseTime + index * 60 * 1000).toISOString();
}

// ── Reusable event builders ──

function userMessage(id: string, text: string, index: number): GatewayEvent {
  return {
    id,
    session_id: sessionId,
    created_at: at(index),
    payload: {
      type: "message",
      role: "user",
      content: [{ type: "text", text }],
      metadata: {},
    },
  };
}

function toolCall(
  tcId: string,
  name: string,
  args: JsonValue,
  index: number,
): GatewayEvent {
  return {
    id: tcId,
    session_id: sessionId,
    created_at: at(index),
    payload: {
      type: "tool_call",
      name,
      arguments: args,
      requires_approval: true,
      side_info: {
        tool_call_event_id: tcId,
        session_id: sessionId,
        config_snapshot_hash: "abc",
        optimization: {
          poll_interval_secs: BigInt(60),
          max_wait_secs: BigInt(86400),
        },
      },
    },
  };
}

function toolAuth(
  eventId: string,
  tcId: string,
  name: string,
  args: JsonValue,
  status: { type: "approved" } | { type: "rejected"; reason: string },
  index: number,
): GatewayEvent {
  return {
    id: eventId,
    session_id: sessionId,
    created_at: at(index),
    payload: {
      type: "tool_call_authorization",
      source: { type: "ui" },
      status,
      tool_call_event_id: tcId,
      tool_call_name: name,
      tool_call_arguments: args,
    },
  };
}

function toolResultSuccess(
  eventId: string,
  tcId: string,
  name: string,
  args: JsonValue,
  result: string,
  index: number,
): GatewayEvent {
  return {
    id: eventId,
    session_id: sessionId,
    created_at: at(index),
    payload: {
      type: "tool_result",
      tool_call_event_id: tcId,
      tool_call_name: name,
      tool_call_arguments: args,
      tool_call_authorization_source: { type: "automatic" as const },
      tool_call_authorization_status: { type: "approved" as const },
      outcome: { type: "success" as const, result },
    },
  };
}

function toolResultFailure(
  eventId: string,
  tcId: string,
  name: string,
  args: JsonValue,
  message: string,
  index: number,
): GatewayEvent {
  return {
    id: eventId,
    session_id: sessionId,
    created_at: at(index),
    payload: {
      type: "tool_result",
      tool_call_event_id: tcId,
      tool_call_name: name,
      tool_call_arguments: args,
      tool_call_authorization_source: { type: "ui" as const },
      tool_call_authorization_status: { type: "approved" as const },
      outcome: {
        type: "failure" as const,
        error: {
          kind: "tool" as const,
          error: { kind: "validation" as const, message },
        },
      },
    },
  };
}

const meta = {
  title: "Autopilot/EventStream/Tool Call",
  component: EventStream,
  decorators: [
    (Story) => (
      <AutopilotSessionProvider>
        <Story />
      </AutopilotSessionProvider>
    ),
  ],
  render: (args) => (
    <div className="w-[80vw] max-w-3xl p-4">
      <EventStream {...args} />
    </div>
  ),
} satisfies Meta<typeof EventStream>;

export default meta;
type Story = StoryObj<typeof meta>;

// ══════════════════════════════════════════════════════════════
// 1. All Variants
// Shows every badge variant side by side. Each event has a unique
// tool_call_event_id so nothing gets superseded.
// ══════════════════════════════════════════════════════════════

export const AllVariants: Story = {
  name: "All Variants",
  args: {
    events: [
      // Pending Approval (tool_call only)
      toolCall("av-tc-pending", "search_docs", { query: "getting started" }, 0),
      // Pending Execution (auth approved, no result)
      toolAuth(
        "av-auth-exec",
        "av-tc-exec",
        "run_migration",
        { version: "v2.3.0" },
        { type: "approved" },
        1,
      ),
      // Success
      toolResultSuccess(
        "av-result-success",
        "av-tc-success",
        "fetch_data",
        { source: "metrics_db" },
        "Retrieved 42 rows from metrics_db.",
        2,
      ),
      // Error
      toolResultFailure(
        "av-result-error",
        "av-tc-error",
        "write_report",
        { format: "pdf" },
        "PDF renderer crashed unexpectedly",
        3,
      ),
      // Rejected (auth rejected)
      toolAuth(
        "av-auth-rejected",
        "av-tc-auth-rej",
        "drop_database",
        { target: "production" },
        { type: "rejected", reason: "Blocked by admin policy" },
        4,
      ),
      // Rejected (result rejected)
      {
        id: "av-result-rejected",
        session_id: sessionId,
        created_at: at(5),
        payload: {
          type: "tool_result" as const,
          tool_call_event_id: "av-tc-result-rej",
          tool_call_name: "delete_records",
          tool_call_arguments: { table: "users" },
          tool_call_authorization_source: { type: "ui" as const },
          tool_call_authorization_status: {
            type: "rejected" as const,
            reason: "Destructive operation",
          },
          outcome: {
            type: "rejected" as const,
            reason: "User declined the action.",
          },
        },
      },
      // Missing Tool
      {
        id: "av-result-missing",
        session_id: sessionId,
        created_at: at(6),
        payload: {
          type: "tool_result" as const,
          tool_call_event_id: "av-tc-missing",
          tool_call_name: "nonexistent_tool",
          tool_call_arguments: {},
          tool_call_authorization_source: { type: "automatic" as const },
          tool_call_authorization_status: { type: "approved" as const },
          outcome: { type: "missing" as const },
        },
      },
      // Unknown
      {
        id: "av-result-unknown",
        session_id: sessionId,
        created_at: at(7),
        payload: {
          type: "tool_result" as const,
          tool_call_event_id: "av-tc-unknown",
          tool_call_name: "future_tool",
          tool_call_arguments: {},
          tool_call_authorization_source: { type: "automatic" as const },
          tool_call_authorization_status: { type: "approved" as const },
          outcome: { type: "unknown" as const },
        },
      },
    ],
  },
};

// ══════════════════════════════════════════════════════════════
// 2. Lifecycle stories
// Each shows a single tool chain at a specific stage.
// Superseded events are present in the data but hidden by the UI.
// ══════════════════════════════════════════════════════════════

// Stage 1: tool_call only → "Pending Approval" badge
export const LifecyclePendingApproval: Story = {
  name: "Lifecycle: Pending Approval",
  args: {
    events: [
      userMessage("lc-pa-user", "Search for deployment docs.", 0),
      toolCall("lc-pa-tc", "search_wikipedia", { query: "deployment docs" }, 1),
    ],
  },
};

// Stage 2: tool_call + approved auth → tool_call hidden, auth shows "Pending Execution"
export const LifecyclePendingExecution: Story = {
  name: "Lifecycle: Pending Execution",
  args: {
    events: [
      userMessage("lc-pe-user", "Run the database migration.", 0),
      toolCall("lc-pe-tc", "run_migration", { version: "v2.3.0" }, 1),
      toolAuth(
        "lc-pe-auth",
        "lc-pe-tc",
        "run_migration",
        { version: "v2.3.0" },
        { type: "approved" },
        2,
      ),
    ],
  },
};

// Stage 3: full chain → success result visible, tool_call + auth hidden
export const LifecycleSuccess: Story = {
  name: "Lifecycle: Success",
  args: {
    events: [
      userMessage("lc-s-user", "Fetch the latest metrics.", 0),
      toolCall("lc-s-tc", "fetch_data", { source: "metrics_db" }, 1),
      toolAuth(
        "lc-s-auth",
        "lc-s-tc",
        "fetch_data",
        { source: "metrics_db" },
        { type: "approved" },
        2,
      ),
      toolResultSuccess(
        "lc-s-result",
        "lc-s-tc",
        "fetch_data",
        { source: "metrics_db" },
        "Retrieved 42 rows from metrics_db.",
        3,
      ),
    ],
  },
};

// Stage 3 (error): full chain → error result visible, tool_call + auth hidden
export const LifecycleError: Story = {
  name: "Lifecycle: Error",
  args: {
    events: [
      userMessage("lc-e-user", "Generate the quarterly report.", 0),
      toolCall("lc-e-tc", "write_report", { format: "pdf" }, 1),
      toolAuth(
        "lc-e-auth",
        "lc-e-tc",
        "write_report",
        { format: "pdf" },
        { type: "approved" },
        2,
      ),
      toolResultFailure(
        "lc-e-result",
        "lc-e-tc",
        "write_report",
        { format: "pdf" },
        "PDF renderer crashed unexpectedly",
        3,
      ),
    ],
  },
};

// Stage 3 (rejected): tool_call + rejected auth → tool_call hidden, auth shows "Rejected"
export const LifecycleRejected: Story = {
  name: "Lifecycle: Rejected",
  args: {
    events: [
      userMessage("lc-r-user", "Drop the production database.", 0),
      toolCall("lc-r-tc", "drop_database", { target: "production" }, 1),
      toolAuth(
        "lc-r-auth",
        "lc-r-tc",
        "drop_database",
        { target: "production" },
        {
          type: "rejected",
          reason:
            "Destructive operations require manual approval in the admin console.",
        },
        2,
      ),
    ],
  },
};

// Whitelisted: tool_call + result with no auth event → tool_call hidden, result visible
export const LifecycleWhitelisted: Story = {
  name: "Lifecycle: Whitelisted",
  args: {
    events: [
      userMessage("lc-w-user", "Look up the current deployment config.", 0),
      toolCall("lc-w-tc", "read_config", { section: "deployment" }, 1),
      toolResultSuccess(
        "lc-w-result",
        "lc-w-tc",
        "read_config",
        { section: "deployment" },
        "Current deployment: region=us-east-1, replicas=3",
        2,
      ),
    ],
  },
};

// ══════════════════════════════════════════════════════════════
// 3. Superseding / multiple chains
// Demonstrates that tool chains with different tool_call_event_ids
// are independent — one chain's events don't hide another's.
// ══════════════════════════════════════════════════════════════

// Four independent tools at different lifecycle stages in a single stream.
// Each chain supersedes only its own earlier events.
// Expected visible events:
//   - User message
//   - tool_a_pending (tool_call, "Pending Approval")
//   - tool_b_authorized (auth, "Pending Execution") — tool_call hidden
//   - tool_c_completed (result, "Success") — tool_call + auth hidden
//   - tool_d_failed (result, "Error") — tool_call + auth hidden
export const IndependentChainsMixedStages: Story = {
  name: "Independent Chains: Mixed Stages",
  args: {
    events: [
      userMessage(
        "ic-user",
        "Run four tools: one pending, one approved, one completed, one failed.",
        0,
      ),
      // Tool A: pending approval (tool_call only → visible)
      toolCall("ic-tc-a", "search_wikipedia", { query: "pending search" }, 1),
      // Tool B: approved, awaiting execution (tool_call hidden by auth)
      toolCall("ic-tc-b", "run_migration", { version: "v3.0.0" }, 2),
      toolAuth(
        "ic-auth-b",
        "ic-tc-b",
        "run_migration",
        { version: "v3.0.0" },
        { type: "approved" },
        3,
      ),
      // Tool C: completed successfully (tool_call + auth hidden by result)
      toolCall("ic-tc-c", "fetch_data", { source: "analytics" }, 4),
      toolAuth(
        "ic-auth-c",
        "ic-tc-c",
        "fetch_data",
        { source: "analytics" },
        { type: "approved" },
        5,
      ),
      toolResultSuccess(
        "ic-result-c",
        "ic-tc-c",
        "fetch_data",
        { source: "analytics" },
        "Fetched 128 records from analytics.",
        6,
      ),
      // Tool D: completed with error (tool_call + auth hidden by result)
      toolCall("ic-tc-d", "write_report", { format: "csv" }, 7),
      toolAuth(
        "ic-auth-d",
        "ic-tc-d",
        "write_report",
        { format: "csv" },
        { type: "approved" },
        8,
      ),
      toolResultFailure(
        "ic-result-d",
        "ic-tc-d",
        "write_report",
        { format: "csv" },
        "Disk quota exceeded",
        9,
      ),
    ],
  },
};

// Multiple tools all completed — shows several result rows side by side.
// This is the typical view after the agent has finished a multi-step plan.
export const IndependentChainsAllCompleted: Story = {
  name: "Independent Chains: All Completed",
  args: {
    events: [
      userMessage("ic-ac-user", "Run a full audit of the system.", 0),
      // Chain 1: search
      toolCall("ic-ac-tc-1", "search_wikipedia", { query: "system audit" }, 1),
      toolResultSuccess(
        "ic-ac-result-1",
        "ic-ac-tc-1",
        "search_wikipedia",
        { query: "system audit" },
        "Found 12 relevant articles on system auditing best practices.",
        2,
      ),
      // Chain 2: fetch
      toolCall(
        "ic-ac-tc-2",
        "fetch_data",
        { source: "audit_logs", limit: 100 },
        3,
      ),
      toolResultSuccess(
        "ic-ac-result-2",
        "ic-ac-tc-2",
        "fetch_data",
        { source: "audit_logs", limit: 100 },
        "Retrieved 100 audit log entries spanning the last 30 days.",
        4,
      ),
      // Chain 3: write
      toolCall(
        "ic-ac-tc-3",
        "write_report",
        { format: "markdown", title: "Audit Summary" },
        5,
      ),
      toolResultSuccess(
        "ic-ac-result-3",
        "ic-ac-tc-3",
        "write_report",
        { format: "markdown", title: "Audit Summary" },
        "Report saved to /reports/audit-2026-04.md",
        6,
      ),
    ],
  },
};

// Mixed outcomes — completed chains with both successes and failures.
export const IndependentChainsMixedOutcomes: Story = {
  name: "Independent Chains: Mixed Outcomes",
  args: {
    events: [
      userMessage(
        "ic-mo-user",
        "Search the docs, then try to deploy and clean up.",
        0,
      ),
      // Chain 1: success
      toolCall("ic-mo-tc-1", "search_docs", { query: "deployment guide" }, 1),
      toolResultSuccess(
        "ic-mo-result-1",
        "ic-mo-tc-1",
        "search_docs",
        { query: "deployment guide" },
        "Found deployment guide at /docs/deploy.md",
        2,
      ),
      // Chain 2: rejected by user
      toolCall(
        "ic-mo-tc-2",
        "deploy_to_production",
        { branch: "main", region: "us-east-1" },
        3,
      ),
      toolAuth(
        "ic-mo-auth-2",
        "ic-mo-tc-2",
        "deploy_to_production",
        { branch: "main", region: "us-east-1" },
        { type: "rejected", reason: "Staging tests have not passed yet." },
        4,
      ),
      // Chain 3: error
      toolCall(
        "ic-mo-tc-3",
        "cleanup_temp_files",
        { directory: "/tmp/build" },
        5,
      ),
      toolResultFailure(
        "ic-mo-result-3",
        "ic-mo-tc-3",
        "cleanup_temp_files",
        { directory: "/tmp/build" },
        "Permission denied: /tmp/build is owned by root",
        6,
      ),
      // Chain 4: missing tool
      {
        id: "ic-mo-result-4",
        session_id: sessionId,
        created_at: at(7),
        payload: {
          type: "tool_result" as const,
          tool_call_event_id: "ic-mo-tc-4",
          tool_call_name: "send_notification",
          tool_call_arguments: { channel: "slack" },
          tool_call_authorization_source: { type: "automatic" as const },
          tool_call_authorization_status: { type: "approved" as const },
          outcome: { type: "missing" as const },
        },
      },
    ],
  },
};
