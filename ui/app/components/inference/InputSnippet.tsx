import type {
  ResolvedInput,
  ResolvedInputMessageContent,
  ResolvedInputMessage,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetHeading,
  SnippetDivider,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import {
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  ImageErrorMessage,
  TextMessage,
  EmptyMessage,
} from "~/components/layout/SnippetContent";

interface InputSnippetProps {
  input: ResolvedInput;
}

function renderContentBlock(block: ResolvedInputMessageContent, index: number) {
  switch (block.type) {
    case "text": {
      if (typeof block.value === "object") {
        return (
          <TextMessage
            key={index}
            label="Text (Arguments)"
            content={JSON.stringify(block.value, null, 2)}
            type="structured"
          />
        );
      }

      // Try to parse JSON strings
      if (typeof block.value === "string") {
        try {
          const parsedJson = JSON.parse(block.value);
          if (typeof parsedJson === "object") {
            return (
              <TextMessage
                key={index}
                label="Text (Arguments)"
                content={JSON.stringify(parsedJson, null, 2)}
                type="structured"
              />
            );
          }
        } catch {
          // Not valid JSON, continue with regular text message
        }
      }

      return <TextMessage key={index} label="Text" content={block.value} />;
    }

    case "raw_text":
      return (
        <TextMessage
          key={index}
          label="Text (Raw)"
          content={block.value}
          type="structured"
        />
      );

    case "tool_call":
      return (
        <ToolCallMessage
          key={index}
          toolName={block.name}
          toolArguments={JSON.stringify(JSON.parse(block.arguments), null, 2)}
          // TODO: if arguments is null, display raw arguments without parsing
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
      if (block.file.mime_type.startsWith("image/")) {
        return (
          <ImageMessage
            key={index}
            url={block.file.url}
            downloadName={`tensorzero_${block.storage_path.path}`}
          />
        );
      } else {
        return (
          <div key={index}>
            <ImageErrorMessage
              key={index}
              error={`Unsupported file type: ${block.file.mime_type}`}
            />
          </div>
        );
      }
    case "file_error":
      return <ImageErrorMessage key={index} error="Failed to retrieve image" />;

    default:
      return null;
  }
}

function renderMessage(message: ResolvedInputMessage, messageIndex: number) {
  return (
    <SnippetMessage variant="input" key={messageIndex} role={message.role}>
      {message.content.map(
        (block: ResolvedInputMessageContent, blockIndex: number) =>
          renderContentBlock(block, blockIndex),
      )}
    </SnippetMessage>
  );
}

export default function InputSnippet({ input }: InputSnippetProps) {
  return (
    <SnippetLayout>
      {input.system && (
        <>
          <SnippetHeading heading="System" />
          <SnippetContent>
            <SnippetMessage>
              {typeof input.system === "object" ? (
                <TextMessage
                  content={JSON.stringify(input.system, null, 2)}
                  type="structured"
                />
              ) : (
                <TextMessage content={input.system} />
              )}
            </SnippetMessage>
          </SnippetContent>
          <SnippetDivider />
        </>
      )}

      <div>
        {input.messages.length === 0 ? (
          <SnippetContent>
            <EmptyMessage message="No input messages found" />
          </SnippetContent>
        ) : (
          <>
            <SnippetHeading heading="Messages" />
            <SnippetContent>
              {input.messages.map((message, messageIndex) => (
                <div key={messageIndex}>
                  {renderMessage(message, messageIndex)}
                </div>
              ))}
            </SnippetContent>
          </>
        )}
      </div>
    </SnippetLayout>
  );
}
