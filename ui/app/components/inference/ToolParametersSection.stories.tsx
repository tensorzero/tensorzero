import type { Meta, StoryObj } from "@storybook/react-vite";
import { ToolParametersSection } from "./ToolParametersSection";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
import { Toaster } from "~/components/ui/toaster";

const meta: Meta<typeof ToolParametersSection> = {
  title: "Inference/ToolParametersSection",
  component: ToolParametersSection,
  parameters: {
    layout: "centered",
  },
  decorators: [
    // TODO: CodeEditor has a hard dependency on toast infrastructure via its
    // built-in copy button (CodeEditor -> useCopy -> useToast). This couples
    // a low-level UI component to application-level providers. Consider making
    // the copy feature optional or using inline feedback instead of toasts.
    (Story) => (
      <GlobalToastProvider>
        <div className="w-[600px] p-4">
          <Story />
        </div>
        <Toaster />
      </GlobalToastProvider>
    ),
  ],
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    provider_tools: [],
  },
};

export const WithAllowedTools: Story = {
  args: {
    allowed_tools: ["search_wikipedia", "get_weather", "calculate"],
    provider_tools: [],
  },
};

export const WithToolChoice: Story = {
  args: {
    tool_choice: "auto",
    provider_tools: [],
  },
};

export const WithToolChoiceRequired: Story = {
  args: {
    tool_choice: "required",
    provider_tools: [],
  },
};

export const WithToolChoiceNone: Story = {
  args: {
    tool_choice: "none",
    provider_tools: [],
  },
};

export const WithToolChoiceSpecific: Story = {
  args: {
    tool_choice: { specific: "search_wikipedia" },
    provider_tools: [],
  },
};

export const WithParallelToolCallsEnabled: Story = {
  args: {
    parallel_tool_calls: true,
    provider_tools: [],
  },
};

export const WithParallelToolCallsDisabled: Story = {
  args: {
    parallel_tool_calls: false,
    provider_tools: [],
  },
};

export const WithFunctionTools: Story = {
  args: {
    additional_tools: [
      {
        type: "function",
        name: "search_wikipedia",
        description:
          "Search Wikipedia for pages that match the query. Returns a list of page titles.",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                'The query to search Wikipedia for (e.g. "machine learning").',
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
        strict: true,
      },
      {
        type: "function",
        name: "get_weather",
        description: "Get the current weather for a location.",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and country, e.g. San Francisco, CA",
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
              description: "The temperature unit to use",
            },
          },
          required: ["location"],
        },
        strict: false,
      },
      {
        type: "function",
        name: "calculate",
        description: "Perform a mathematical calculation.",
        parameters: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "The mathematical expression to evaluate",
            },
          },
          required: ["expression"],
        },
        strict: false,
      },
      {
        type: "function",
        name: "send_email_notification",
        description: "Send an email notification to a user.",
        parameters: {
          type: "object",
          properties: {
            to: { type: "string", description: "Recipient email address" },
            subject: { type: "string", description: "Email subject line" },
            body: { type: "string", description: "Email body content" },
          },
          required: ["to", "subject", "body"],
        },
        strict: true,
      },
      {
        type: "function",
        name: "query_database_records",
        description: "Query records from the database with filters.",
        parameters: {
          type: "object",
          properties: {
            table: { type: "string", description: "Table name to query" },
            filters: { type: "object", description: "Filter conditions" },
            limit: { type: "number", description: "Max records to return" },
          },
          required: ["table"],
        },
        strict: false,
      },
      {
        type: "function",
        name: "generate_pdf_report",
        description: "Generate a PDF report from provided data.",
        parameters: {
          type: "object",
          properties: {
            title: { type: "string", description: "Report title" },
            data: { type: "array", description: "Data to include in report" },
            format: { type: "string", enum: ["summary", "detailed"] },
          },
          required: ["title", "data"],
        },
        strict: true,
      },
      {
        type: "function",
        name: "translate_text_content",
        description: "Translate text from one language to another.",
        parameters: {
          type: "object",
          properties: {
            text: { type: "string", description: "Text to translate" },
            source_lang: {
              type: "string",
              description: "Source language code",
            },
            target_lang: {
              type: "string",
              description: "Target language code",
            },
          },
          required: ["text", "target_lang"],
        },
        strict: false,
      },
      {
        type: "function",
        name: "schedule_calendar_event",
        description: "Schedule an event on the calendar.",
        parameters: {
          type: "object",
          properties: {
            title: { type: "string", description: "Event title" },
            start_time: { type: "string", description: "ISO 8601 start time" },
            end_time: { type: "string", description: "ISO 8601 end time" },
            attendees: {
              type: "array",
              description: "List of attendee emails",
            },
          },
          required: ["title", "start_time", "end_time"],
        },
        strict: true,
      },
    ],
    provider_tools: [],
  },
};

export const WithOpenAICustomTool: Story = {
  args: {
    additional_tools: [
      {
        type: "openai_custom",
        name: "code_interpreter",
        description: "Executes Python code and returns the result.",
        format: {
          type: "text",
        },
      },
    ],
    provider_tools: [],
  },
};

export const WithSingleProviderTool: Story = {
  args: {
    provider_tools: [
      {
        scope: null,
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
    ],
  },
};

export const WithMultipleProviderTools: Story = {
  args: {
    provider_tools: [
      {
        scope: { model_name: "gpt-4o", provider_name: "openai" },
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
      {
        scope: { model_name: "claude-sonnet-4-5", provider_name: "anthropic" },
        tool: {
          type: "computer_use",
          display_width: 1920,
          display_height: 1080,
        },
      },
    ],
  },
};

export const WithProviderToolScoped: Story = {
  args: {
    provider_tools: [
      {
        scope: { model_name: "gpt-4o" },
        tool: {
          type: "file_search",
          vector_store_ids: ["vs_123", "vs_456"],
        },
      },
    ],
  },
};

export const Complete: Story = {
  args: {
    allowed_tools: ["search_wikipedia", "get_weather"],
    additional_tools: [
      {
        type: "function",
        name: "calculate",
        description: "Perform a mathematical calculation.",
        parameters: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "The mathematical expression to evaluate",
            },
          },
          required: ["expression"],
        },
        strict: true,
      },
    ],
    tool_choice: "auto",
    parallel_tool_calls: true,
    provider_tools: [
      {
        scope: null,
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
    ],
  },
};
