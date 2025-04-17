import type {
  ResolvedInput,
  ResolvedInputMessageContent,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetHeading,
  SnippetDivider,
} from "~/components/layout/SnippetLayout";
import {
  CodeMessage,
  InputMessage,
  InputTextMessage,
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  ImageErrorMessage,
} from "~/components/layout/SnippetContent";

interface InputSnippetProps {
  input: ResolvedInput;
}

function renderContentBlock(
  block: ResolvedInputMessageContent,
  role: string,
  index: number,
) {
  switch (block.type) {
    case "text": {
      const displayValue =
        typeof block.value === "object"
          ? JSON.stringify(block.value, null, 2)
          : block.value;

      return (
        <InputMessage key={index} role={role}>
          <InputTextMessage content={displayValue} />
        </InputMessage>
      );
    }

    case "tool_call":
      return (
        <InputMessage key={index} role={role}>
          <ToolCallMessage
            label={`Tool: ${block.name}`}
            content={block.arguments}
          />
        </InputMessage>
      );

    case "tool_result":
      return (
        <InputMessage key={index} role={role}>
          <ToolResultMessage
            label={`Result from: ${block.name}`}
            content={block.result}
          />
        </InputMessage>
      );

    case "image":
      return (
        <InputMessage key={index} role={role}>
          <ImageMessage
            url={block.image.url}
            downloadName={`tensorzero_${block.storage_path.path}`}
          />
        </InputMessage>
      );

    case "image_error":
      return (
        <InputMessage key={index} role={role}>
          <ImageErrorMessage />
        </InputMessage>
      );

    default:
      return null;
  }
}

export default function InputSnippet({ input }: InputSnippetProps) {
  return (
    <SnippetLayout>
      {input.system && (
        <div>
          <SnippetHeading heading="System" />
          <SnippetContent>
            <CodeMessage
              content={
                typeof input.system === "object"
                  ? JSON.stringify(input.system, null, 2)
                  : input.system
              }
            />
          </SnippetContent>
          <SnippetDivider />
        </div>
      )}

      <div>
        <SnippetHeading heading="Messages" />
        <SnippetContent>
          <div className="pb-4">
            {input.messages.map((message, messageIndex) => (
              <div key={messageIndex}>
                {message.content.map((block, blockIndex) =>
                  renderContentBlock(block, message.role, blockIndex),
                )}
              </div>
            ))}
          </div>
        </SnippetContent>
      </div>
    </SnippetLayout>
  );
}
