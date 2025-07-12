import type {
  DisplayInputMessageContent,
  DisplayInputMessage,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import {
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  FileErrorMessage,
  FileMessage,
  AudioMessage,
  TextMessage,
  EmptyMessage,
  ParameterizedMessage,
} from "~/components/layout/SnippetContent";
import type { JsonObject } from "type-fest";

interface InputSnippetProps {
  messages: DisplayInputMessage[];
  system?: string | JsonObject | null;
}

function renderContentBlock(block: DisplayInputMessageContent, index: number) {
  switch (block.type) {
    case "structured_text":
      return <ParameterizedMessage key={index} parameters={block.arguments} />;

    // Unstructured text is a function/variant with no schema
    case "unstructured_text":
      return <TextMessage key={index} label="Text" content={block.text} />;

    // "Raw text" is when the user submits an inference on the function/variant and overrides template interpolation
    case "raw_text":
      return <TextMessage key={index} label="Raw Text" content={block.value} />;

    case "missing_function_text":
      return (
        <TextMessage
          key={index}
          label="Text (Missing Function Config)"
          content={block.value}
        />
      );

    case "tool_call":
      return (
        <ToolCallMessage
          key={index}
          toolName={block.name}
          toolRawName={block.name} // tool calls in the input aren't parsed, so there's no "raw"
          toolArguments={block.arguments}
          toolRawArguments={block.arguments} // tool calls in the input aren't parsed, so there's no "raw"
          toolCallId={block.id}
        />
      );

    case "tool_result":
      return (
        <ToolResultMessage
          key={index}
          toolName={block.name}
          toolResult={block.result}
          toolResultId={block.id}
        />
      );

    case "file":
      return block.file.mime_type.startsWith("image/") ? (
        <ImageMessage
          key={index}
          url={block.file.dataUrl}
          downloadName={`tensorzero_${block.storage_path.path}`}
        />
      ) : block.file.mime_type.startsWith("audio/") ? (
        <AudioMessage
          key={index}
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      ) : (
        <FileMessage
          key={index}
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      );

    case "file_error":
      return <FileErrorMessage key={index} error="Failed to retrieve file" />;
  }
}

export default function InputSnippet({ system, messages }: InputSnippetProps) {
  return (
    <SnippetLayout>
      {!system && messages.length === 0 && (
        <SnippetContent>
          <EmptyMessage message="Empty input" />
        </SnippetContent>
      )}

      {system && (
        <SnippetContent>
          <SnippetMessage role="system">
            {typeof system === "object" ? (
              <ParameterizedMessage parameters={system} />
            ) : (
              <TextMessage content={system} />
            )}
          </SnippetMessage>
        </SnippetContent>
      )}

      {messages.length > 0 && (
        <SnippetContent>
          {messages.map((message, messageIndex) => (
            <SnippetMessage role={message.role} key={messageIndex}>
              {message.content.map(renderContentBlock)}
            </SnippetMessage>
          ))}
        </SnippetContent>
      )}
    </SnippetLayout>
  );
}
