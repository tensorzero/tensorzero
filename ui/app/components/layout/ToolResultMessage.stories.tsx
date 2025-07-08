import type { Meta, StoryObj } from "@storybook/react-vite";
import { ToolResultMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/ToolResultMessage",
  component: ToolResultMessage,
  parameters: {
    layout: "padded",
  },
  decorators: [
    (Story) => (
      <div className="max-w-md">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ToolResultMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    toolName: "get_weather",
    toolResult: JSON.stringify(
      {
        location: "San Francisco, CA",
        temperature: 18,
        condition: "Partly Cloudy",
        humidity: 65,
      },
      null,
      2,
    ),
    toolResultId: "call_abc123def456",
  },
};

export const ComplexResult: Story = {
  args: {
    toolName: "search_database",
    toolResult: JSON.stringify(
      {
        results: [
          {
            id: 1,
            name: "Alice Johnson",
            department: "engineering",
            role: "senior",
            join_date: "2021-03-15",
          },
          {
            id: 2,
            name: "Bob Smith",
            department: "engineering",
            role: "lead",
            join_date: "2020-08-22",
          },
        ],
        total_count: 47,
        has_more: true,
        execution_time_ms: 156,
      },
      null,
      2,
    ),
    toolResultId: "call_xyz789abc123",
  },
};

export const ErrorResult: Story = {
  args: {
    toolName: "file_processor",
    toolResult: JSON.stringify(
      {
        success: false,
        error: {
          code: "FILE_NOT_FOUND",
          message: "The specified file could not be located in the system",
          details: {
            path: "/uploads/documents/report.pdf",
            timestamp: "2024-01-15T10:30:00Z",
          },
        },
      },
      null,
      2,
    ),
    toolResultId: "call_error_example_123",
  },
};

export const EmptyResult: Story = {
  args: {
    toolName: "clear_cache",
    toolResult: JSON.stringify(
      {
        success: true,
        message: "Cache cleared successfully",
      },
      null,
      2,
    ),
    toolResultId: "call_simple_123",
  },
};

export const LargeDataResult: Story = {
  args: {
    toolName: "fetch_metrics",
    toolResult: JSON.stringify(
      {
        metrics: Array.from({ length: 10 }, (_, i) => ({
          timestamp: `2024-01-15T${String(i).padStart(2, "0")}:00:00Z`,
          cpu: Math.random() * 100,
          memory: Math.random() * 8192,
          requests: Math.floor(Math.random() * 1000),
        })),
        summary: {
          avg_cpu: 45.2,
          avg_memory: 4096,
          total_requests: 5432,
        },
      },
      null,
      2,
    ),
    toolResultId: "call_metrics_456",
  },
};
