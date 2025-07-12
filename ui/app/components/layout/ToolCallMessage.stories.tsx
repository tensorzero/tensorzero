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

const simpletoolArguments = JSON.stringify(
  {
    location: "San Francisco, CA",
    units: "celsius",
  },
  null,
  2,
);

export const Simple: Story = {
  args: {
    toolName: "get_weather",
    toolRawName: "get_weather",
    toolArguments: simpletoolArguments,
    toolRawArguments: simpletoolArguments,
    toolCallId: "call_abc123def456",
  },
};

const complexToolArguments = JSON.stringify(
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
);

export const ComplexArguments: Story = {
  args: {
    toolName: "search_database",
    toolRawName: "search_database",
    toolArguments: complexToolArguments,
    toolRawArguments: complexToolArguments,
    toolCallId: "call_xyz789abc123",
  },
};

export const LongToolName: Story = {
  args: {
    toolName: "generate_quarterly_financial_report_with_charts_and_analysis",
    toolRawName: "generate_quarterly_financial_report_with_charts_and_analysis",
    toolArguments: simpletoolArguments,
    toolRawArguments: simpletoolArguments,
    toolCallId: "call_abc123",
  },
};

export const LongToolId: Story = {
  args: {
    toolName: "simple_tool",
    toolRawName: "simple_tool",
    toolArguments: simpletoolArguments,
    toolRawArguments: simpletoolArguments,
    toolCallId:
      "call_very_long_tool_call_id_that_should_be_truncated_properly_12345678901234567890",
  },
};

export const MinimalArguments: Story = {
  args: {
    toolName: "ping",
    toolRawName: "ping",
    toolArguments: "{}",
    toolRawArguments: "{}",
    toolCallId: "call_123",
  },
};
