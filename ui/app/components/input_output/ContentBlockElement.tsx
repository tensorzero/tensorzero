import { type ReactNode } from "react";
import type { InputMessageContent } from "~/types/tensorzero";
import { TextContentBlock } from "~/components/input_output/content_blocks/TextContentBlock";
import { TemplateContentBlock } from "~/components/input_output/content_blocks/TemplateContentBlock";
import { ToolCallContentBlock } from "~/components/input_output/content_blocks/ToolCallContentBlock";
import { ToolResultContentBlock } from "~/components/input_output/content_blocks/ToolResultContentBlock";
import { FileContentBlock } from "~/components/input_output/content_blocks/FileContentBlock";
import { ThoughtContentBlock } from "~/components/input_output/content_blocks/ThoughtContentBlock";
import { UnknownContentBlock } from "~/components/input_output/content_blocks/UnknownContentBlock";

interface ContentBlockElementProps {
  block: InputMessageContent;
  isEditing?: boolean;
  onChange?: (updatedContentBlock: InputMessageContent) => void;
  actionBar?: ReactNode;
}

/**
 * Renders a content block based on its type
 * Supports both input and output content blocks
 */
export function ContentBlockElement({
  block,
  isEditing,
  onChange,
  actionBar,
}: ContentBlockElementProps): ReactNode {
  switch (block.type) {
    case "text": {
      return (
        <TextContentBlock
          label="Text"
          text={block.text}
          isEditing={isEditing}
          onChange={(updatedText) => {
            onChange?.({ ...block, text: updatedText });
          }}
          actionBar={actionBar}
        />
      );
    }

    case "raw_text": {
      return (
        <TextContentBlock
          label="Raw Text"
          text={block.value}
          isEditing={isEditing}
          onChange={(updatedValue) => {
            onChange?.({ ...block, value: updatedValue });
          }}
          actionBar={actionBar}
        />
      );
    }

    case "tool_call": {
      return (
        <ToolCallContentBlock
          block={block}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedBlock) => {
                  onChange?.({
                    ...block,
                    ...updatedBlock,
                  });
                }
              : undefined
          }
          actionBar={actionBar}
        />
      );
    }

    case "tool_result": {
      return (
        <ToolResultContentBlock
          block={block}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedBlock) => {
                  onChange?.({
                    ...block,
                    ...updatedBlock,
                  });
                }
              : undefined
          }
          actionBar={actionBar}
        />
      );
    }

    case "file": {
      return (
        <FileContentBlock
          block={block}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedBlock) => {
                  onChange?.({
                    ...block,
                    ...updatedBlock,
                  });
                }
              : undefined
          }
          actionBar={actionBar}
        />
      );
    }

    case "unknown": {
      return (
        <UnknownContentBlock
          data={block.data}
          isEditing={isEditing}
          onChange={(data) =>
            onChange?.({ ...block, data: data as typeof block.data })
          }
          actionBar={actionBar}
        />
      );
    }

    case "thought": {
      return (
        <ThoughtContentBlock
          block={block}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedBlock) => {
                  onChange?.({
                    ...block,
                    ...updatedBlock,
                  });
                }
              : undefined
          }
          actionBar={actionBar}
        />
      );
    }

    case "template": {
      return (
        <TemplateContentBlock
          block={block}
          isEditing={isEditing}
          onChange={
            isEditing
              ? (updatedContentBlock) => {
                  onChange?.({
                    ...block,
                    ...updatedContentBlock,
                  });
                }
              : undefined
          }
          actionBar={actionBar}
        />
      );
    }
  }
}

export function EmptyMessage({ message = "No content" }: { message?: string }) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
      {message}
    </div>
  );
}
