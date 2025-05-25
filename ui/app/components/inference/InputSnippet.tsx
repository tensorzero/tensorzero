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
  CodeMessage,
  InputTextMessage,
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  ImageErrorMessage,
  TextMessageWithArguments,
  RawTextMessage,
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
        return <TextMessageWithArguments key={index} content={block.value} />;
      }

      // Try to parse JSON strings
      if (typeof block.value === "string") {
        try {
          const parsedJson = JSON.parse(block.value);
          if (typeof parsedJson === "object") {
            return (
              <TextMessageWithArguments key={index} content={parsedJson} />
            );
          }
        } catch {
          // Not valid JSON, continue with regular text message
        }
      }

      return <InputTextMessage key={index} content={block.value} />;
    }

    case "raw_text":
      return <RawTextMessage key={index} content={block.value} />;

    case "tool_call":
      return (
        <ToolCallMessage
          key={index}
          toolName={block.name}
          toolArguments={JSON.stringify(block.arguments, null, 2)}
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
            <ImageErrorMessage key={index} error={`Unsupported file type: ${block.file.mime_type}`} />
          </div>
        )
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
        <div>
          <SnippetHeading heading="System" />
          <SnippetContent>
            <SnippetMessage>
              {typeof input.system === "object" ? (
                <CodeMessage
                  content={JSON.stringify(input.system, null, 2)}
                  showLineNumbers={true}
                />
              ) : (
                <TextMessage content={input.system} />
              )}
            </SnippetMessage>
          </SnippetContent>
          <SnippetDivider />
        </div>
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
              <div className="pb-4">
                {input.messages.map((message, messageIndex) => (
                  <div key={messageIndex}>
                    {renderMessage(message, messageIndex)}
                  </div>
                ))}
              </div>
            </SnippetContent>
          </>
        )}
      </div>
    </SnippetLayout>
  );
}
