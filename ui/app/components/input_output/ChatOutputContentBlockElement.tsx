import { type ReactNode } from "react";
import type { ContentBlockChatOutput } from "~/types/tensorzero";
import { TextContentBlock } from "~/components/input_output/content_blocks/TextContentBlock";
import { InferenceResponseToolCallContentBlock } from "~/components/input_output/content_blocks/InferenceResponseToolCallContentBlock";
import { ThoughtContentBlock } from "~/components/input_output/content_blocks/ThoughtContentBlock";

interface ChatOutputContentBlockElementProps {
  block: ContentBlockChatOutput;
  isEditing?: boolean;
  onChange?: (updatedContentBlock: ContentBlockChatOutput) => void;
  actionBar?: ReactNode;
}

/**
 * Renders a content block based on its type
 */
export function ChatOutputContentBlockElement({
  block,
  isEditing,
  onChange,
  actionBar,
}: ChatOutputContentBlockElementProps): ReactNode {
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

    case "tool_call": {
      return (
        <InferenceResponseToolCallContentBlock
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
        <TextContentBlock
          label="Unknown"
          text={JSON.stringify(block.data)}
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
  }
}

export function EmptyMessage({ message = "No content" }: { message?: string }) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
      {message}
    </div>
  );
}
