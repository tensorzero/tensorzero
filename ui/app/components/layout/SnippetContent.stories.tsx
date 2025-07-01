import type { Meta, StoryObj } from "@storybook/react-vite";
import {
  EmptyMessage,
  TextMessage,
  CodeMessage,
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  FileErrorMessage,
  AudioMessage,
  FileMessage,
} from "./SnippetContent";

// Sample data for file components
const samplePdfData = "data:application/pdf;base64,JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKL01lZGlhQm94IFswIDAgNTk1IDg0Ml0KPj4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA0IDAgUgo+Pgo+PgovTWVkaWFCb3ggWzAgMCA1OTUgODQyXQovQ29udGVudHMgNSAwIFIKPj4KZW5kb2JqCjQgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iago1IDAgb2JqCjw8Ci9MZW5ndGggNDQKPj4Kc3RyZWFtCkJUCi9GMSAxMiBUZgo3MiA3MjAgVGQKKEhlbGxvLCB3b3JsZCEpVGoKRVQKZW5kc3RyZWFtCmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAyNDUgMDAwMDAgbiAKMDAwMDAwMDMxNCAwMDAwMCBuIAp0cmFpbGVyCjw8Ci9TaXplIDYKL1Jvb3QgMSAwIFIKPj4Kc3RhcnR4cmVmCjQwOAolJUVPRg==";

const sampleAudioData = "data:audio/mp3;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV6urq6urq6urq6urq6urq6urq6urq6urq6v////////////////////////////////8AAAAATGF2YzU4LjU0AAAAAAAAAAAAAAAAJAAAAAAAAAAAASDs90hvAAAAAAAAAAAAAAAAAAAA//sQxAADwAABpAAAACAAADSAAAAETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQ==";

const sampleImageData = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=";

// EmptyMessage Stories
const emptyMessageMeta = {
  title: "SnippetContent/EmptyMessage",
  component: EmptyMessage,
  parameters: {
    layout: "centered",
  },
} satisfies Meta<typeof EmptyMessage>;

export default emptyMessageMeta;
type EmptyMessageStory = StoryObj<typeof emptyMessageMeta>;

export const DefaultEmpty: EmptyMessageStory = {
  args: {},
};

export const CustomEmptyMessage: EmptyMessageStory = {
  args: {
    message: "No data available at this time",
  },
};

// TextMessage Stories
const textMessageMeta = {
  title: "SnippetContent/TextMessage",
  component: TextMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof TextMessage>;

export const DefaultText: StoryObj<typeof textMessageMeta> = {
  args: {
    content: "This is a simple text message that demonstrates how content is displayed.",
  },
};

export const TextWithLabel: StoryObj<typeof textMessageMeta> = {
  args: {
    label: "User Input",
    content: "This text message includes a label to provide context about the content.",
  },
};

export const StructuredTextMessage: StoryObj<typeof textMessageMeta> = {
  args: {
    label: "Structured Content",
    content: `{
  "topic": "Machine Learning",
  "style": "academic",
  "length": "detailed",
  "include_examples": true
}`,
    type: "structured",
  },
};

export const LongTextMessage: StoryObj<typeof textMessageMeta> = {
  args: {
    label: "Long Content",
    content: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
  },
};

export const EmptyTextMessage: StoryObj<typeof textMessageMeta> = {
  args: {
    label: "Empty Content",
    content: "",
    emptyMessage: "No text content provided",
  },
};

// CodeMessage Stories
const codeMessageMeta = {
  title: "SnippetContent/CodeMessage",
  component: CodeMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof CodeMessage>;

export const SimpleCode: StoryObj<typeof codeMessageMeta> = {
  args: {
    content: `function greet(name) {
  return \`Hello, \${name}!\`;
}

console.log(greet("World"));`,
  },
};

export const CodeWithLabel: StoryObj<typeof codeMessageMeta> = {
  args: {
    label: "JavaScript Function",
    content: `function calculateFactorial(n) {
  if (n <= 1) return 1;
  return n * calculateFactorial(n - 1);
}`,
  },
};

export const CodeWithLineNumbers: StoryObj<typeof codeMessageMeta> = {
  args: {
    label: "Python Script",
    content: `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate first 10 fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")`,
    showLineNumbers: true,
  },
};

export const MultilineCodeWithLineNumbers: StoryObj<typeof codeMessageMeta> = {
  args: {
    label: "React Component",
    content: `import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}

export default Counter;`,
    showLineNumbers: true,
  },
};

export const EmptyCodeMessage: StoryObj<typeof codeMessageMeta> = {
  args: {
    label: "Empty Code",
    content: "",
    emptyMessage: "No code content available",
  },
};

// ToolCallMessage Stories
const toolCallMessageMeta = {
  title: "SnippetContent/ToolCallMessage",
  component: ToolCallMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof ToolCallMessage>;

export const SimpleToolCall: StoryObj<typeof toolCallMessageMeta> = {
  args: {
    toolName: "get_weather",
    toolArguments: JSON.stringify({
      location: "San Francisco, CA",
      units: "celsius",
    }),
    toolCallId: "call_abc123def456",
  },
};

export const ComplexToolCall: StoryObj<typeof toolCallMessageMeta> = {
  args: {
    toolName: "search_database",
    toolArguments: JSON.stringify({
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
    }),
    toolCallId: "call_xyz789abc123",
  },
};

export const LongToolCallId: StoryObj<typeof toolCallMessageMeta> = {
  args: {
    toolName: "generate_report_with_very_long_name",
    toolArguments: JSON.stringify({
      report_type: "quarterly_financial_summary",
      include_charts: true,
    }),
    toolCallId: "call_very_long_tool_call_id_that_should_be_truncated_properly_12345678901234567890",
  },
};

// ToolResultMessage Stories
const toolResultMessageMeta = {
  title: "SnippetContent/ToolResultMessage",
  component: ToolResultMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof ToolResultMessage>;

export const SimpleToolResult: StoryObj<typeof toolResultMessageMeta> = {
  args: {
    toolName: "get_weather",
    toolResult: JSON.stringify({
      location: "San Francisco, CA",
      temperature: 18,
      condition: "Partly Cloudy",
      humidity: 65,
    }),
    toolResultId: "call_abc123def456",
  },
};

export const ComplexToolResult: StoryObj<typeof toolResultMessageMeta> = {
  args: {
    toolName: "search_database",
    toolResult: JSON.stringify({
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
    }),
    toolResultId: "call_xyz789abc123",
  },
};

export const ErrorToolResult: StoryObj<typeof toolResultMessageMeta> = {
  args: {
    toolName: "file_processor",
    toolResult: JSON.stringify({
      success: false,
      error: {
        code: "FILE_NOT_FOUND",
        message: "The specified file could not be located in the system",
        details: {
          path: "/uploads/documents/report.pdf",
          timestamp: "2024-01-15T10:30:00Z",
        },
      },
    }),
    toolResultId: "call_error_example_123",
  },
};

// ImageMessage Stories
const imageMessageMeta = {
  title: "SnippetContent/ImageMessage",
  component: ImageMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof ImageMessage>;

export const SimpleImage: StoryObj<typeof imageMessageMeta> = {
  args: {
    url: sampleImageData,
  },
};

export const ImageWithDownloadName: StoryObj<typeof imageMessageMeta> = {
  args: {
    url: sampleImageData,
    downloadName: "sample_image.png",
  },
};

// FileErrorMessage Stories
const fileErrorMessageMeta = {
  title: "SnippetContent/FileErrorMessage",
  component: FileErrorMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof FileErrorMessage>;

export const FileNotFound: StoryObj<typeof fileErrorMessageMeta> = {
  args: {
    error: "File not found: The requested file could not be located",
  },
};

export const NetworkError: StoryObj<typeof fileErrorMessageMeta> = {
  args: {
    error: "Network timeout: Failed to download file after 30 seconds",
  },
};

export const CorruptedFile: StoryObj<typeof fileErrorMessageMeta> = {
  args: {
    error: "File corrupted: The file appears to be damaged and cannot be displayed",
  },
};

export const LongErrorMessage: StoryObj<typeof fileErrorMessageMeta> = {
  args: {
    error: "Authentication failed: The access token has expired and the system was unable to refresh it automatically. Please log in again to continue accessing this resource.",
  },
};

// AudioMessage Stories
const audioMessageMeta = {
  title: "SnippetContent/AudioMessage",
  component: AudioMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof AudioMessage>;

export const MP3Audio: StoryObj<typeof audioMessageMeta> = {
  args: {
    fileData: sampleAudioData,
    filePath: "sample_audio.mp3",
    mimeType: "audio/mp3",
  },
};

export const LongFilenameAudio: StoryObj<typeof audioMessageMeta> = {
  args: {
    fileData: sampleAudioData,
    filePath: "very_long_audio_filename_that_should_be_truncated_properly_in_the_display.mp3",
    mimeType: "audio/mpeg",
  },
};

// FileMessage Stories
const fileMessageMeta = {
  title: "SnippetContent/FileMessage",
  component: FileMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof FileMessage>;

export const PDFFile: StoryObj<typeof fileMessageMeta> = {
  args: {
    fileData: samplePdfData,
    filePath: "document.pdf",
    mimeType: "application/pdf",
  },
};

export const TextFile: StoryObj<typeof fileMessageMeta> = {
  args: {
    fileData: "data:text/plain;base64,VGhpcyBpcyBhIHNhbXBsZSB0ZXh0IGZpbGU=",
    filePath: "notes.txt",
    mimeType: "text/plain",
  },
};

export const LongFilenameFile: StoryObj<typeof fileMessageMeta> = {
  args: {
    fileData: samplePdfData,
    filePath: "very_long_document_filename_that_should_be_truncated_properly_in_the_file_display_component.pdf",
    mimeType: "application/pdf",
  },
};

export const JSONFile: StoryObj<typeof fileMessageMeta> = {
  args: {
    fileData: "data:application/json;base64,eyJuYW1lIjoiSm9obiBEb2UiLCJhZ2UiOjMwLCJjaXR5IjoiTmV3IFlvcmsifQ==",
    filePath: "config.json",
    mimeType: "application/json",
  },
};