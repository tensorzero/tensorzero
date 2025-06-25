import type {
  DisplayInput,
  DisplayInputMessageContent,
  DisplayInputMessage,
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
  FileErrorMessage,
  FileMessage,
  AudioMessage,
  TextMessage,
  EmptyMessage,
} from "~/components/layout/SnippetContent";

interface InputSnippetProps {
  input: DisplayInput;
}

function renderContentBlock(block: DisplayInputMessageContent, index: number) {
  switch (block.type) {
    case "structured_text": {
      return (
        <TextMessage
          key={index}
          label="Text (Arguments)"
          content={JSON.stringify(block.arguments, null, 2)}
          type="structured"
        />
      );
    }

    case "unstructured_text": {
      return (
        <TextMessage
          key={index}
          label="Text"
          content={block.text}
          type="default"
        />
      );
    }

    case "missing_function_text": {
      return (
        <TextMessage
          key={index}
          label="Text (Missing Function Config)"
          content={block.value}
          type="default"
        />
      );
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

    case "tool_call": {
      let serializedArguments;
      try {
        serializedArguments = JSON.stringify(
          JSON.parse(block.arguments),
          null,
          2,
        );
      } catch {
        serializedArguments = block.arguments;
      }
      return (
        <ToolCallMessage
          key={index}
          toolName={block.name}
          toolArguments={serializedArguments}
          // TODO: if arguments is null, display raw arguments without parsing
          toolCallId={block.id}
        />
      );
    }

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
            url={block.file.dataUrl}
            downloadName={`tensorzero_${block.storage_path.path}`}
          />
        );
      } else if (block.file.mime_type.startsWith("audio/")) {
        return (
          <AudioMessage
            key={index}
            fileData={block.file.dataUrl}
            mimeType={block.file.mime_type}
            filePath={block.storage_path.path}
          />
        );
      } else {
        return (
          <FileMessage
            key={index}
            fileData={block.file.dataUrl}
            mimeType={block.file.mime_type}
            filePath={block.storage_path.path}
          />
        );
      }

    case "file_error":
      return <FileErrorMessage key={index} error="Failed to retrieve file" />;

    default:
      return null;
  }
}

function renderMessage(message: DisplayInputMessage, messageIndex: number) {
  return (
    <SnippetMessage variant="input" key={messageIndex} role={message.role}>
      {message.content.map(
        (block: DisplayInputMessageContent, blockIndex: number) =>
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
                (() => {
                  let serializedSystem;
                  try {
                    serializedSystem = JSON.stringify(input.system, null, 2);
                  } catch {
                    return null;
                  }
                  return (
                    <TextMessage content={serializedSystem} type="structured" />
                  );
                })()
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
