import type { Meta, StoryObj } from "@storybook/react-vite";
import { ToolCallMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/ToolCallMessage",
  component: ToolCallMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof ToolCallMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    toolName: "get_weather",
    toolArguments: JSON.stringify(
      {
        location: "San Francisco, CA",
        units: "celsius",
      },
      null,
      2,
    ),
    toolCallId: "call_abc123def456",
  },
};

export const ComplexArguments: Story = {
  args: {
    toolName: "search_database",
    toolArguments: JSON.stringify(
      {
        query: "SELECT * FROM users WHERE active = true",
        filters: {
          department: "engineering",
          role: ["senior", "lead"],
          join_date: {
            after: "2020-01-01",
            before: "2023-12-31",
          },
        },
        pagination: {
          limit: 50,
          offset: 0,
        },
        sort: [
          { field: "last_name", direction: "asc" },
          { field: "join_date", direction: "desc" },
        ],
      },
      null,
      2,
    ),
    toolCallId: "call_xyz789abc123",
  },
};

export const LongToolName: Story = {
  args: {
    toolName: "generate_quarterly_financial_report_with_charts_and_analysis",
    toolArguments: JSON.stringify(
      {
        report_type: "quarterly_financial_summary",
        include_charts: true,
      },
      null,
      2,
    ),
    toolCallId: "call_abc123",
  },
};

export const LongToolId: Story = {
  args: {
    toolName: "simple_tool",
    toolArguments: JSON.stringify(
      {
        param: "value",
      },
      null,
      2,
    ),
    toolCallId:
      "call_very_long_tool_call_id_that_should_be_truncated_properly_12345678901234567890",
  },
};

export const MinimalArguments: Story = {
  args: {
    toolName: "ping",
    toolArguments: "{}",
    toolCallId: "call_123",
  },
};
