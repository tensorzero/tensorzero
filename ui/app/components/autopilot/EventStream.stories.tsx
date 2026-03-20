import type { Meta, StoryObj } from "@storybook/react-vite";
import EventStream from "./EventStream";
import type { GatewayEvent } from "~/types/tensorzero";
import { AutopilotSessionProvider } from "~/contexts/AutopilotSessionContext";

const baseTime = new Date("2026-04-12T10:00:00Z").getTime();
const sessionId = "d1a0b0c0-0000-0000-0000-000000000001";

function buildEvent(event: GatewayEvent, index: number): GatewayEvent {
  return {
    ...event,
    created_at: new Date(baseTime + index * 60 * 1000).toISOString(),
  };
}

const conversationEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "e2a3f5d6-7b8c-4d9e-8f01-1234567890a1",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "Summarize the deployment status." }],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "f3b4c5d6-7e8f-4a1b-9c2d-2345678901b2",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Deployments are healthy across all regions.",
          },
        ],
        metadata: {},
      },
    },
    1,
  ),
];

const toolingEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "0a1b2c3d-4e5f-4a6b-8c7d-3456789012c3",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: { query: "TensorZero" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "0a1b2c3d-4e5f-4a6b-8c7d-3456789012c3",
          session_id: sessionId,
          config_snapshot_hash: "hi",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "1b2c3d4e-5f6a-4b7c-8d9e-4567890123d4",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "0a1b2c3d-4e5f-4a6b-8c7d-3456789012c3",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "TensorZero" },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "2c3d4e5f-6a7b-4c8d-9e0f-5678901234e5",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "0a1b2c3d-4e5f-4a6b-8c7d-3456789012c3",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "TensorZero" },
        tool_call_authorization_source: { type: "ui" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "Found relevant context.",
        },
      },
    },
    2,
  ),
];

const mixedEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "Run a quick session audit." }],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "3d4e5f6a-7b8c-4d9e-8f01-6789012345f6",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "status_update",
        status_update: { type: "text", text: "Syncing metadata" },
      },
    },
    1,
  ),
  // Chain 1: tool_call -> authorization (approved) -> result (success)
  buildEvent(
    {
      id: "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: { query: "TensorZero Autopilot" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
          session_id: sessionId,
          config_snapshot_hash: "hi",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "c3d4e5f6-7a8b-4c9d-8e0f-2a3b4c5d6e7f",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "TensorZero Autopilot" },
      },
    },
    3,
  ),
  buildEvent(
    {
      id: "d4e5f6a7-8b9c-4d0e-9f1a-3b4c5d6e7f80",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "TensorZero Autopilot" },
        tool_call_authorization_source: { type: "ui" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "Found audit notes on recent Autopilot sessions.",
        },
      },
    },
    4,
  ),
  // Chain 2: authorization (rejected) -> result (failure) - tool_call not shown
  buildEvent(
    {
      id: "4e5f6a7b-8c9d-4e0f-9a1b-7890123456a7",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "rejected", reason: "Blocked by policy" },
        tool_call_event_id: "rejected-tool-call-event-id",
        tool_call_name: "dangerous_tool",
        tool_call_arguments: {},
      },
    },
    5,
  ),
  buildEvent(
    {
      id: "5f6a7b8c-9d0e-4f1a-8b2c-8901234567b8",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "rejected-tool-call-event-id",
        tool_call_name: "dangerous_tool",
        tool_call_arguments: {},
        tool_call_authorization_source: { type: "ui" as const },
        tool_call_authorization_status: {
          type: "rejected" as const,
          reason: "Blocked by policy",
        },
        outcome: {
          type: "failure",
          error: {
            kind: "tool",
            error: { kind: "validation", message: "Authorization denied" },
          },
        },
      },
    },
    6,
  ),
  // Standalone tool_result with missing outcome
  buildEvent(
    {
      id: "f6a7b8c9-0d1e-4f2a-8b3c-4d5e6f708192",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "missing-tool-call-event-id",
        tool_call_name: "missing_tool",
        tool_call_arguments: {},
        tool_call_authorization_source: { type: "automatic" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "missing",
        },
      },
    },
    7,
  ),
  // Standalone tool_result with other outcome
  buildEvent(
    {
      id: "a7b8c9d0-1e2f-4a3b-9c4d-5e6f708192a3",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "other-tool-call-event-id",
        tool_call_name: "other_tool",
        tool_call_arguments: {},
        tool_call_authorization_source: { type: "automatic" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "unknown",
        },
      },
    },
    8,
  ),
  buildEvent(
    {
      id: "e5f6a7b8-9c0d-4e1f-8a2b-4c5d6e7f8091",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Audit complete. One policy block detected and resolved.",
          },
        ],
        metadata: {},
      },
    },
    9,
  ),
  buildEvent(
    {
      id: "6a7b8c9d-0e1f-4a2b-9c3d-9012345678c9",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "unknown",
      },
    },
    10,
  ),
];

const longText =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
  "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. " +
  "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. " +
  "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

const markdownText = `# Project Overview

This is a **markdown** example with various formatting options.

## Features
- Bullet point one
- Bullet point two
- Bullet point three

### Code Example
\`\`\`javascript
function hello() {
  console.log("Hello, world!");
}
\`\`\`

### Links and Formatting
Visit [TensorZero](https://tensorzero.com) for more information.

*Italic text* and **bold text** are supported.

> This is a blockquote with important information.

1. Ordered list item
2. Another item
3. Final item`;

const markdownEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "md-user-1",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          { type: "text", text: "Can you give me a project overview?" },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "md-assistant-1",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: markdownText }],
        metadata: {},
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "md-user-2",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "Thanks! Can you show inline code?" }],
        metadata: {},
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "md-assistant-2",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Sure! You can use `console.log()` to print values, or run `npm install` to install dependencies.",
          },
        ],
        metadata: {},
      },
    },
    3,
  ),
];

const longToolArguments = JSON.stringify({
  query: longText,
  filters: {
    tags: ["autopilot", "tensorzero", "stream", "events", "tooling"],
    include_archived: false,
  },
});

const longToolResult = [
  "Section A:",
  longText,
  "Section B:",
  longText,
  "Section C:",
  longText,
].join("\n\n");

const longFormEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "7b8c9d0e-1f2a-4b3c-8d4e-0123456789d0",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: longText }],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "8c9d0e1f-2a3b-4c4d-9e5f-1234567890e1",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: JSON.parse(longToolArguments),
        requires_approval: true,
        side_info: {
          tool_call_event_id: "8c9d0e1f-2a3b-4c4d-9e5f-1234567890e1",
          session_id: sessionId,
          config_snapshot_hash: "hi",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "9d0e1f2a-3b4c-4d5e-8f6a-2345678901f2",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "8c9d0e1f-2a3b-4c4d-9e5f-1234567890e1",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: JSON.parse(longToolArguments),
        tool_call_authorization_source: { type: "automatic" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: longToolResult,
        },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "0e1f2a3b-4c5d-4e6f-9a7b-3456789012a3",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: `${longText}\n\n${longText}`,
          },
        ],
        metadata: {},
      },
    },
    3,
  ),
];

const meta = {
  title: "Autopilot/EventStream",
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

export const Empty: Story = {
  args: {
    events: [],
  },
};

export const Conversation: Story = {
  args: {
    events: conversationEvents,
  },
};

export const Tooling: Story = {
  args: {
    events: toolingEvents,
  },
};

export const Mixed: Story = {
  args: {
    events: mixedEvents,
  },
};

export const LongForm: Story = {
  args: {
    events: longFormEvents,
  },
};

// Events demonstrating visualization rendering with tool results
const visualizationEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "v1-user-msg",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Run a top-k evaluation on the test variants.",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "v2-tool-call",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "topk_evaluation",
        requires_approval: true,
        arguments: {
          evaluation_name: "test_topk_evaluation",
          dataset_name: "topk_test_dataset",
          variant_names: ["echo", "empty", "empty2", "test", "test2"],
          k_min: 1,
          max_datapoints: 100,
        },
        side_info: {
          tool_call_event_id: "v2-tool-call",
          session_id: sessionId,
          config_snapshot_hash: "abc123",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "v3-auth",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "v2-tool-call",
        tool_call_name: "topk_evaluation",
        tool_call_arguments: {
          evaluation_name: "test_topk_evaluation",
          dataset_name: "topk_test_dataset",
          variant_names: ["echo", "empty", "empty2", "test", "test2"],
          k_min: 1,
          max_datapoints: 100,
        },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "v4-result",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "v2-tool-call",
        tool_call_name: "topk_evaluation",
        tool_call_arguments: {
          evaluation_name: "test_topk_evaluation",
          dataset_name: "topk_test_dataset",
          variant_names: ["echo", "empty", "empty2", "test", "test2"],
          k_min: 1,
          max_datapoints: 100,
        },
        tool_call_authorization_source: { type: "ui" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: JSON.stringify(
            {
              winner: "echo",
              final_k: 3,
              evaluations_run: 150,
            },
            null,
            2,
          ),
        },
      },
    },
    3,
  ),
  // Visualization event tied to the tool call
  // Data designed to show separation lines at top-1 and top-3:
  // Non-failed variants sorted by lower bound: echo(0.85) > empty2(0.70) > empty(0.60) > test(0.45)
  // Top-1: echo's lower (0.85) > max upper of rest (0.80) ✓ (clear gap, line at midpoint)
  // Top-3: empty's lower (0.60) < test's upper (0.62) - overlap due to epsilon tolerance
  //        (line drawn below empty's LCB to test epsilon-based separation)
  // Failed variants (displayed on right): failed_bad_variant1, failed_bad_variant2
  buildEvent(
    {
      id: "v5-visualization",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "visualization",
        tool_execution_id: "v2-tool-call",
        visualization: {
          type: "top_k_evaluation",
          variant_summaries: {
            echo: {
              mean_est: 0.9,
              cs_lower: 0.85,
              cs_upper: 0.95,
              count: BigInt(50),
              failed: false,
            },
            test: {
              mean_est: 0.52,
              cs_lower: 0.45,
              cs_upper: 0.62, // UCB overlaps with empty's LCB (0.6) to test epsilon-based separation
              count: BigInt(25),
              failed: false,
            },
            empty: {
              mean_est: 0.65,
              cs_lower: 0.6,
              cs_upper: 0.7,
              count: BigInt(45),
              failed: false,
            },
            empty2: {
              mean_est: 0.75,
              cs_lower: 0.7,
              cs_upper: 0.8,
              count: BigInt(35),
              failed: false,
            },
            failed_bad_variant1: {
              mean_est: 0.55,
              cs_lower: 0.45,
              cs_upper: 0.65,
              count: BigInt(20),
              failed: true,
            },
            failed_bad_variant2: {
              mean_est: 0.32,
              cs_lower: 0.25,
              cs_upper: 0.4,
              count: BigInt(15),
              failed: true,
            },
          },
          confident_top_k_sizes: [1, 3],
          summary_text: `## Overview
The top-k evaluation tool identifies the best-performing variants from a set of candidates using adaptive evaluation with statistical confidence bounds.

## Chart Description
The top chart shows the estimated mean performance with final 95% confidence intervals for each variant.

The bottom chart shows the number of evaluations per variant.

## Results
The algorithm identified the top-1 variant and the top-3 variants.

## Efficiency
- Total evaluations: 190.
- Total evaluations that would have been run without adaptive stopping: 300.
- Savings: (300 - 190) / 300 * 100% = 36.7%.

## Next Steps
Deploy the "echo" variant to production for improved performance.`,
        },
      },
    },
    4,
  ),
  buildEvent(
    {
      id: "v6-assistant-msg",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: 'The top-k evaluation is complete. We can confidently identify "echo" as the top-1 performer (mean: 0.90) and a top-3 set of "echo", "test", and "empty" that statistically outperforms the remaining variants.',
          },
        ],
        metadata: {},
      },
    },
    5,
  ),
];

export const WithVisualization: Story = {
  args: {
    events: visualizationEvents,
  },
};

const unknownVisualizationEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "uv1-user-msg",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Run an analysis on the data.",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "uv2-visualization",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "visualization",
        tool_execution_id: "some-task-id",
        // Unknown visualization type - simulates a future visualization type
        // that the current UI doesn't know how to render
        visualization: {
          type: "future_analysis",
          data: {
            metric_a: 0.95,
            metric_b: 0.87,
            samples: 1000,
          },
          metadata: {
            version: "2.0",
            algorithm: "advanced_analysis",
          },
        } as GatewayEvent["payload"] extends { visualization: infer V }
          ? V
          : never,
      },
    },
    1,
  ),
];

export const WithUnknownVisualization: Story = {
  args: {
    events: unknownVisualizationEvents,
  },
};

// Events demonstrating a whitelisted tool call that skips the authorization step
const whitelistedToolingEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "wt-user-msg",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          { type: "text", text: "Look up the current deployment config." },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "wt-tool-call",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "read_config",
        arguments: { section: "deployment" },
        requires_approval: false,
        side_info: {
          tool_call_event_id: "wt-tool-call",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  // No authorization event — whitelisted tools go straight to result
  buildEvent(
    {
      id: "wt-tool-result",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "wt-tool-call",
        tool_call_name: "read_config",
        tool_call_arguments: { section: "deployment" },
        tool_call_authorization_source: { type: "whitelist" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "Current deployment: region=us-east-1, replicas=3",
        },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "wt-assistant-msg",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "The current deployment is running in us-east-1 with 3 replicas.",
          },
        ],
        metadata: {},
      },
    },
    3,
  ),
];

export const WhitelistedTooling: Story = {
  args: {
    events: whitelistedToolingEvents,
  },
};

export const MarkdownContent: Story = {
  args: {
    events: markdownEvents,
  },
};

// ── Stories covering event superseding / collapsing logic ──

// Scenario: tool_call exists but no auth or result yet → tool_call is visible
const pendingToolCallEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "pending-tc-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [{ type: "text", text: "Run a search for me." }],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "pending-tc",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: { query: "pending search" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "pending-tc",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
];

export const SupersedingPendingToolCall: Story = {
  name: "Superseding / Pending Tool Call (visible)",
  args: {
    events: pendingToolCallEvents,
  },
};

// Scenario: tool_call + auth (no result yet) → tool_call hidden, auth visible
const authOnlyEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "auth-only-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Run a search (authorized, waiting for result).",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "auth-only-tc",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: { query: "auth only search" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "auth-only-tc",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "auth-only-auth",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "auth-only-tc",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "auth only search" },
      },
    },
    2,
  ),
];

export const SupersedingAuthOnly: Story = {
  name: "Superseding / Auth hides Tool Call (no result yet)",
  args: {
    events: authOnlyEvents,
  },
};

// Scenario: full chain tool_call + auth + result → only result visible
const fullChainEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "full-chain-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Run a search (full chain — only result should appear).",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "full-chain-tc",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "search_wikipedia",
        arguments: { query: "full chain search" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "full-chain-tc",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "full-chain-auth",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "full-chain-tc",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "full chain search" },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "full-chain-result",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "full-chain-tc",
        tool_call_name: "search_wikipedia",
        tool_call_arguments: { query: "full chain search" },
        tool_call_authorization_source: { type: "ui" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "Found relevant results for full chain search.",
        },
      },
    },
    3,
  ),
];

export const SupersedingFullChain: Story = {
  name: "Superseding / Full chain (result hides tool_call + auth)",
  args: {
    events: fullChainEvents,
  },
};

// Scenario: whitelisted tool — tool_call + result (no auth event) → tool_call hidden
const whitelistedSupersedingEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "wl-sup-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Read config (whitelisted — tool_call hidden by result, no auth event).",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "wl-sup-tc",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "read_config",
        arguments: { section: "network" },
        requires_approval: false,
        side_info: {
          tool_call_event_id: "wl-sup-tc",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "wl-sup-result",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "wl-sup-tc",
        tool_call_name: "read_config",
        tool_call_arguments: { section: "network" },
        tool_call_authorization_source: { type: "whitelist" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "network: port=8080, host=0.0.0.0",
        },
      },
    },
    2,
  ),
];

export const SupersedingWhitelistedTool: Story = {
  name: "Superseding / Whitelisted tool (result hides tool_call, no auth)",
  args: {
    events: whitelistedSupersedingEvents,
  },
};

// Scenario: multiple tool chains at different stages in one stream
const mixedSupersedingEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "mix-sup-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Three tools at different stages: pending, authorized, completed.",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  // Tool A: pending (no auth, no result) → visible
  buildEvent(
    {
      id: "mix-tc-a",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "tool_a_pending",
        arguments: { step: "pending" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "mix-tc-a",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    1,
  ),
  // Tool B: authorized (tool_call hidden, auth visible)
  buildEvent(
    {
      id: "mix-tc-b",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "tool_b_authorized",
        arguments: { step: "authorized" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "mix-tc-b",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    2,
  ),
  buildEvent(
    {
      id: "mix-auth-b",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        status: { type: "approved" },
        tool_call_event_id: "mix-tc-b",
        tool_call_name: "tool_b_authorized",
        tool_call_arguments: { step: "authorized" },
      },
    },
    3,
  ),
  // Tool C: completed (tool_call + auth hidden, result visible)
  buildEvent(
    {
      id: "mix-tc-c",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call",
        name: "tool_c_completed",
        arguments: { step: "completed" },
        requires_approval: true,
        side_info: {
          tool_call_event_id: "mix-tc-c",
          session_id: sessionId,
          config_snapshot_hash: "abc",
          optimization: {
            poll_interval_secs: BigInt(60),
            max_wait_secs: BigInt(86400),
          },
        },
      },
    },
    4,
  ),
  buildEvent(
    {
      id: "mix-auth-c",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_call_authorization",
        source: { type: "automatic" },
        status: { type: "approved" },
        tool_call_event_id: "mix-tc-c",
        tool_call_name: "tool_c_completed",
        tool_call_arguments: { step: "completed" },
      },
    },
    5,
  ),
  buildEvent(
    {
      id: "mix-result-c",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "tool_result",
        tool_call_event_id: "mix-tc-c",
        tool_call_name: "tool_c_completed",
        tool_call_arguments: { step: "completed" },
        tool_call_authorization_source: { type: "automatic" as const },
        tool_call_authorization_status: { type: "approved" as const },
        outcome: {
          type: "success",
          result: "Tool C completed successfully.",
        },
      },
    },
    6,
  ),
];

export const SupersedingMixedStages: Story = {
  name: "Superseding / Mixed stages (pending, authorized, completed)",
  args: {
    events: mixedSupersedingEvents,
  },
};

// Scenario: auto_eval_example_labeling superseded by answers
const autoEvalExampleLabelingEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "ael-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Label these examples (labeling event should be hidden once answers exist).",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "ael-labeling",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "auto_eval_example_labeling",
        examples: [
          {
            maybe_excerpted_prompt: {
              type: "markdown",
              text: "What is TensorZero?",
              label: "Prompt",
            },
            maybe_excerpted_response: {
              type: "markdown",
              text: "TensorZero is an ML optimization platform.",
              label: "Response",
            },
            source: {
              type: "synthetic",
              full_prompt: {
                type: "markdown",
                text: "What is TensorZero?",
              },
              full_response: {
                type: "markdown",
                text: "TensorZero is an ML optimization platform.",
              },
            },
            label_question: {
              id: "q1",
              header: "Quality",
              question: "Is this response accurate?",
              options: [
                {
                  id: "yes",
                  label: "Yes",
                  description: "The response is accurate",
                },
                {
                  id: "no",
                  label: "No",
                  description: "The response is inaccurate",
                },
              ],
            },
          },
        ],
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "ael-answers",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "auto_eval_example_labeling_answers",
        auto_eval_example_labeling_event_id: "ael-labeling",
        examples: [
          {
            maybe_excerpted_prompt: {
              type: "markdown",
              text: "What is TensorZero?",
              label: "Prompt",
            },
            maybe_excerpted_response: {
              type: "markdown",
              text: "TensorZero is an ML optimization platform.",
              label: "Response",
            },
            source: {
              type: "synthetic",
              full_prompt: {
                type: "markdown",
                text: "What is TensorZero?",
              },
              full_response: {
                type: "markdown",
                text: "TensorZero is an ML optimization platform.",
              },
            },
            label_question: {
              id: "q1",
              header: "Quality",
              question: "Is this response accurate?",
              options: [
                {
                  id: "yes",
                  label: "Yes",
                  description: "The response is accurate",
                },
                {
                  id: "no",
                  label: "No",
                  description: "The response is inaccurate",
                },
              ],
            },
            label_answer: {
              type: "multiple_choice",
              selected: ["yes"],
            },
          },
        ],
      },
    },
    2,
  ),
];

export const SupersedingAutoEvalExampleLabeling: Story = {
  name: "Superseding / Auto eval example labeling (answers hide labeling)",
  args: {
    events: autoEvalExampleLabelingEvents,
  },
};

// Scenario: auto_eval_behavior_spec superseded by answers
const autoEvalBehaviorSpecEvents: GatewayEvent[] = [
  buildEvent(
    {
      id: "abs-user",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: "Define behavior spec (spec event should be hidden once answers exist).",
          },
        ],
        metadata: {},
      },
    },
    0,
  ),
  buildEvent(
    {
      id: "abs-spec",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "auto_eval_behavior_spec",
        target_behavior: {
          id: "tb1",
          header: "Target Behavior",
          question: "What should the model do?",
          default_value: "Respond accurately and concisely.",
        },
        additional_context: {
          id: "ac1",
          header: "Additional Context",
          question: "Any extra context for evaluation?",
          default_value: "Focus on factual accuracy.",
        },
      },
    },
    1,
  ),
  buildEvent(
    {
      id: "abs-answers",
      session_id: sessionId,
      created_at: "",
      payload: {
        type: "auto_eval_behavior_spec_answers",
        auto_eval_behavior_spec_event_id: "abs-spec",
        target_behavior: {
          id: "tb1",
          header: "Target Behavior",
          question: "What should the model do?",
          default_value: "Respond accurately and concisely.",
        },
        target_behavior_answer: {
          type: "free_response",
          text: "Respond accurately and concisely.",
        },
        additional_context: {
          id: "ac1",
          header: "Additional Context",
          question: "Any extra context for evaluation?",
          default_value: "Focus on factual accuracy.",
        },
        additional_context_answer: {
          type: "free_response",
          text: "Focus on factual accuracy.",
        },
      },
    },
    2,
  ),
];

export const SupersedingAutoEvalBehaviorSpec: Story = {
  name: "Superseding / Auto eval behavior spec (answers hide spec)",
  args: {
    events: autoEvalBehaviorSpecEvents,
  },
};
