import type { Meta, StoryObj } from "@storybook/react-vite";
import EventStream from "./EventStream";
import type { GatewayEvent } from "~/types/tensorzero";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
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
        outcome: {
          type: "failure",
          error: { kind: "Validation", message: "Authorization denied" },
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
        <GlobalToastProvider>
          <Story />
        </GlobalToastProvider>
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
  // Non-failed variants sorted by lower bound: echo(0.85) > empty2(0.70) > empty(0.60) > test(0.35)
  // Top-1: echo's lower (0.85) > max upper of rest (0.80) ✓
  // Top-3: empty's lower (0.60) > test's upper (0.50) ✓
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
              mean_est: 0.42,
              cs_lower: 0.35,
              cs_upper: 0.5,
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
          summary_text: `## Top-K Evaluation Overview
The top-k evaluation tool identifies the best-performing variants from a set of candidates using adaptive evaluation with statistical confidence bounds.

## Chart Descriptions
The top chart shows the estimated mean performance with final 95% confidence intervals for each variant.

The bottom chart shows the number of evaluations per variant.

## Results
The algorithm identified the top-1 variant and the top-3 variants.

### Efficiency
- Total evaluations: 170.
- Total evaluations that would have been run without adaptive stopping: 250.
- Savings: (250 - 170) / 250 * 100% = 32%.

### Recommended Next Steps
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

export const MarkdownContent: Story = {
  args: {
    events: markdownEvents,
  },
};
