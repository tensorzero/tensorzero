import type { ReactNode } from "react";
import type { DisplayInputMessageContent } from "~/utils/clickhouse/common";
import type { ContentBlockChatOutput } from "tensorzero-node";
import {
  ToolCallMessage,
  ToolResultMessage,
  ImageMessage,
  FileErrorMessage,
  FileMessage,
  AudioMessage,
  TextMessage,
  TemplateMessage,
} from "~/components/layout/SnippetContent";

/**
 * Union type for all content blocks that can be rendered
 */
export type RenderableContentBlock =
  | DisplayInputMessageContent
  | ContentBlockChatOutput;

/**
 * Props for the ContentBlockRenderer component
 */
interface ContentBlockRendererProps {
  block: RenderableContentBlock;
  isEditing?: boolean;
  onChange?: (updatedContentBlock: RenderableContentBlock) => void;
  action?: ReactNode;
  // Optional footer for thought blocks (used by Output.tsx)
  thoughtFooter?: (
    block: Extract<ContentBlockChatOutput, { type: "thought" }>,
  ) => ReactNode;
}

/**
 * Renders a content block based on its type
 * Supports both input and output content blocks
 */
export function ContentBlockRenderer({
  block,
  isEditing,
  onChange,
  action,
  thoughtFooter,
}: ContentBlockRendererProps): ReactNode {
  switch (block.type) {
    // Text content (unstructured text from function/variant with no schema)
    case "text": {
      return (
        <TextMessage
          label="Text"
          content={block.text}
          isEditing={isEditing}
          onChange={(updatedText) => {
            onChange?.({ ...block, text: updatedText });
          }}
          action={action}
        />
      );
    }

    // "Raw text" is when the user submits an inference on the function/variant and overrides template interpolation
    case "raw_text": {
      return (
        <TextMessage
          label="Raw Text"
          content={block.value}
          isEditing={isEditing}
          onChange={(updatedValue) => {
            onChange?.({ ...block, value: updatedValue });
          }}
          action={action}
        />
      );
    }

    case "missing_function_text": {
      return (
        <TextMessage
          label="Text (Missing Function Config)"
          content={block.value}
          isEditing={isEditing}
          onChange={(updatedValue) => {
            onChange?.({ ...block, value: updatedValue });
          }}
          action={action}
        />
      );
    }

    case "tool_call": {
      // Handle both input and output tool calls
      // Input: has `arguments` as string, `name`, `id`
      // Output: has `arguments` as parsed JSON, `raw_arguments` as string, `name`, `raw_name`, `id`
      const isOutputBlock = "raw_arguments" in block;

      if (isOutputBlock) {
        // Output tool call - explicitly narrow the type
        const outputBlock = block as Extract<
          ContentBlockChatOutput,
          { type: "tool_call" }
        >;

        const toolName = outputBlock.name;
        const toolArguments = outputBlock.arguments
          ? JSON.stringify(outputBlock.arguments, null, 2)
          : null;
        const toolRawName = outputBlock.raw_name;
        const toolRawArguments = outputBlock.raw_arguments;

        return (
          <ToolCallMessage
            toolName={toolName}
            toolRawName={toolRawName}
            toolArguments={toolArguments}
            toolRawArguments={toolRawArguments}
            toolCallId={outputBlock.id}
            isEditing={isEditing}
            onChange={
              isEditing
                ? (toolCallId, toolName, toolArguments) => {
                    try {
                      const parsedArgs = JSON.parse(toolArguments);
                      onChange?.({
                        ...outputBlock,
                        id: toolCallId,
                        name: toolName,
                        raw_name: toolName,
                        arguments: parsedArgs,
                        raw_arguments: toolArguments,
                      });
                    } catch {
                      // If parsing fails, set arguments to null but keep raw_arguments
                      onChange?.({
                        ...outputBlock,
                        id: toolCallId,
                        name: null,
                        raw_name: toolName,
                        arguments: null,
                        raw_arguments: toolArguments,
                      });
                    }
                  }
                : undefined
            }
            action={action}
          />
        );
      } else {
        // Input tool call
        const inputBlock = block as Extract<
          DisplayInputMessageContent,
          { type: "tool_call" }
        >;

        return (
          <ToolCallMessage
            toolName={inputBlock.name}
            toolRawName={inputBlock.name}
            toolArguments={inputBlock.arguments}
            toolRawArguments={inputBlock.arguments}
            toolCallId={inputBlock.id}
            isEditing={isEditing}
            onChange={
              isEditing
                ? (toolCallId, toolName, toolArguments) => {
                    onChange?.({
                      ...inputBlock,
                      id: toolCallId,
                      name: toolName,
                      arguments: toolArguments,
                    });
                  }
                : undefined
            }
            action={action}
          />
        );
      }
    }

    case "tool_result": {
      return (
        <ToolResultMessage
          toolName={block.name}
          toolResult={block.result}
          toolResultId={block.id}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (id, name, result) => {
                  onChange?.({ ...block, id, name, result });
                }
              : undefined
          }
          action={action}
        />
      );
    }

    case "file": {
      return block.file.mime_type.startsWith("image/") ? (
        <ImageMessage
          url={block.file.dataUrl}
          downloadName={`tensorzero_${block.storage_path.path}`}
        />
      ) : block.file.mime_type.startsWith("audio/") ? (
        <AudioMessage
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      ) : (
        <FileMessage
          fileData={block.file.dataUrl}
          mimeType={block.file.mime_type}
          filePath={block.storage_path.path}
        />
      );
    }

    case "file_error": {
      return <FileErrorMessage error="Failed to retrieve file" />;
    }

    case "unknown": {
      return (
        <TextMessage
          label="Unknown Content"
          content={JSON.stringify(block.data)}
          action={action}
        />
      );
    }

    case "thought": {
      return (
        <TextMessage
          label="Thought"
          content={block.text || ""}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedText) => {
                  onChange?.({ ...block, text: updatedText });
                }
              : undefined
          }
          footer={thoughtFooter?.(block)}
          action={action}
        />
      );
    }

    case "template": {
      return (
        <TemplateMessage
          templateName={block.name}
          templateArguments={block.arguments}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedName, updatedArguments) => {
                  onChange?.({
                    ...block,
                    name: updatedName,
                    arguments: updatedArguments,
                  });
                }
              : undefined
          }
          action={action}
        />
      );
    }
  }
}
