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
  TemplateMessage,
} from "~/components/layout/SnippetContent";
import type { JsonObject } from "type-fest";
import { Button } from "~/components/ui/button";
import { Plus, Trash2 } from "lucide-react";

interface InputSnippetProps {
  messages: DisplayInputMessage[];
  system?: string | JsonObject | null;
  isEditing?: boolean;
  onSystemChange?: (system: string | object | null) => void;
  onMessagesChange?: (messages: DisplayInputMessage[]) => void;
  maxHeight?: number | "Content";
}

function renderContentBlock(
  block: DisplayInputMessageContent,
  key: string,
  isEditing?: boolean,
  onChange?: (updatedContentBlock: DisplayInputMessageContent) => void,
) {
  switch (block.type) {
    // Unstructured text is a function/variant with no schema
    case "text":
      return (
        <TextMessage
          key={key}
          label="Text"
          content={block.text}
          isEditing={isEditing}
          onChange={(updatedText) => {
            onChange?.({ ...block, text: updatedText });
          }}
        />
      );

    // "Raw text" is when the user submits an inference on the function/variant and overrides template interpolation
    case "raw_text":
      return (
        <TextMessage
          key={key}
          label="Raw Text"
          content={block.value}
          isEditing={isEditing}
          onChange={(updatedValue) => {
            onChange?.({ ...block, value: updatedValue });
          }}
        />
      );

    case "missing_function_text":
      return (
        <TextMessage
          key={key}
          label="Text (Missing Function Config)"
          content={block.value}
          isEditing={isEditing}
          onChange={(updatedValue) => {
            onChange?.({ ...block, value: updatedValue });
          }}
        />
      );

    case "tool_call":
      // NOTE: since tool calls are stored as a string in ResolvedInput and therefore the database
      // and we are not guaranteed that they are valid JSON, we try to parse them as JSON
      // and if they are not valid JSON, we display the raw string
      return (
        <ToolCallMessage
          key={key}
          toolName={block.name}
          toolRawName={block.name} // tool calls in the input aren't parsed, so there's no "raw"
          toolArguments={block.arguments}
          toolRawArguments={block.arguments} // tool calls in the input aren't parsed, so there's no "raw"
          toolCallId={block.id}
          isEditing={isEditing}
          onChange={(toolCallId, toolName, toolArguments) => {
            onChange?.({
              ...block,
              id: toolCallId,
              name: toolName,
              arguments: toolArguments,
            });
          }}
        />
      );

    case "tool_result":
      return (
        <ToolResultMessage
          key={key}
          toolName={block.name}
          toolResult={block.result}
          toolResultId={block.id}
          isEditing={isEditing}
          onChange={(id, name, result) => {
            onChange?.({ ...block, id, name, result });
          }}
        />
      );

    case "file":
      return block.file.mime_type.startsWith("image/") ? (
        <ImageMessage
          key={key}
          url={block.file.dataUrl}
          downloadName={`tensorzero_${block.storage_path.path}`}
        />
      ) : block.file.mime_type.startsWith("audio/") ? (
        <AudioMessage
          key={key}
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      ) : (
        <FileMessage
          key={key}
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      );

    case "file_error":
      return <FileErrorMessage key={key} error="Failed to retrieve file" />;

    case "unknown":
      // TODO: code editor should format as JSON by default
      return (
        <TextMessage
          key={key}
          label="Unknown Content"
          content={JSON.stringify(block.data)}
        />
      );

    case "thought":
      return (
        <TextMessage
          key={key}
          label="Thought"
          content={block.text || ""}
          isEditing={isEditing}
          onChange={(updatedText) => {
            onChange?.({ ...block, text: updatedText });
          }}
        />
      );

    case "template":
      return (
        <TemplateMessage
          key={key}
          templateName={block.name}
          arguments={block.arguments}
          isEditing={isEditing}
          onChange={(updatedArguments) => {
            onChange?.({ ...block, arguments: updatedArguments });
          }}
        />
      );
  }
}

export default function InputSnippet({
  system,
  messages,
  isEditing,
  onSystemChange,
  onMessagesChange,
  maxHeight,
}: InputSnippetProps) {
  const onContentBlockChange = (
    messageIndex: number,
    contentBlockIndex: number,
    updatedContentBlock: DisplayInputMessageContent,
  ) => {
    const updatedMessages = [...messages];
    const updatedMessage = { ...updatedMessages[messageIndex] };
    const updatedContent = [...updatedMessage.content];
    updatedContent[contentBlockIndex] = updatedContentBlock;
    updatedMessage.content = updatedContent;
    updatedMessages[messageIndex] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  return (
    <SnippetLayout>
      {system == null && messages.length === 0 && !isEditing && (
        <SnippetContent maxHeight={maxHeight}>
          <EmptyMessage message="Empty input" />
        </SnippetContent>
      )}

      {system == null && isEditing && (
        <div className="flex items-center py-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onSystemChange?.("")}
            className="flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            Add System
          </Button>
        </div>
      )}

      {system != null && (
        <SnippetContent maxHeight={maxHeight}>
          <SnippetMessage
            role="system"
            action={
              isEditing ? (
                <Button
                  variant="outline"
                  size="iconSm"
                  onClick={() => onSystemChange?.(null)}
                  aria-label="Delete system message"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              ) : undefined
            }
          >
            {typeof system === "object" ? (
              <TemplateMessage
                arguments={system}
                templateName="system"
                isEditing={isEditing}
                onChange={onSystemChange}
              />
            ) : (
              <TextMessage
                content={system}
                isEditing={isEditing}
                onChange={onSystemChange}
              />
            )}
          </SnippetMessage>
        </SnippetContent>
      )}

      {messages.length > 0 && (
        <SnippetContent maxHeight={maxHeight}>
          {messages.map((message, messageIndex) => (
            <SnippetMessage role={message.role} key={messageIndex}>
              {message.content.map((block, contentBlockIndex) =>
                renderContentBlock(
                  block,
                  `${messageIndex}-${contentBlockIndex}`,
                  isEditing,
                  (updatedContentBlock) =>
                    onContentBlockChange(
                      messageIndex,
                      contentBlockIndex,
                      updatedContentBlock,
                    ),
                ),
              )}
            </SnippetMessage>
          ))}
        </SnippetContent>
      )}
    </SnippetLayout>
  );
}
