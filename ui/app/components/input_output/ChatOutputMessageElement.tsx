import type { ContentBlockChatOutput } from "~/types/tensorzero";
import { ChatOutputContentBlockElement } from "./ChatOutputContentBlockElement";
import { MessageWrapper } from "./MessageWrapper";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";

export function ChatOutputMessageElement({
  output,
  isEditing,
  onChange,
  onDelete,
}: {
  output: ContentBlockChatOutput[];
  isEditing?: boolean;
  onChange?: (updatedOutput: ContentBlockChatOutput[]) => void;
  onDelete?: () => void;
}) {
  const onAppendContentBlock = (contentBlock: ContentBlockChatOutput) => {
    const updatedOutput = [...output, contentBlock];
    onChange?.(updatedOutput);
  };

  const onUpdateContentBlock = (
    contentBlockIndex: number,
    updatedContentBlock: ContentBlockChatOutput,
  ) => {
    const updatedOutput = [...output];
    updatedOutput[contentBlockIndex] = updatedContentBlock;
    onChange?.(updatedOutput);
  };

  const onDeleteContentBlock = (contentBlockIndex: number) => {
    const updatedOutput = [...output];
    updatedOutput.splice(contentBlockIndex, 1);
    onChange?.(updatedOutput);
  };

  return (
    <MessageWrapper
      role={"assistant"}
      actionBar={
        isEditing && <DeleteButton onDelete={onDelete} label="Delete output" />
      }
    >
      {output.length === 0 && !isEditing && (
        <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
          Empty message
        </div>
      )}
      {output.map((block, contentBlockIndex) => (
        <ChatOutputContentBlockElement
          key={contentBlockIndex}
          block={block}
          isEditing={isEditing}
          onChange={(updatedContentBlock) =>
            onUpdateContentBlock(contentBlockIndex, updatedContentBlock)
          }
          actionBar={
            isEditing && (
              <DeleteButton
                onDelete={() => onDeleteContentBlock(contentBlockIndex)}
                label="Delete content block"
              />
            )
          }
        />
      ))}
      {isEditing && <AddContentBlockButtons onAdd={onAppendContentBlock} />}
    </MessageWrapper>
  );
}

function AddContentBlockButtons({
  onAdd,
}: {
  onAdd: (block: ContentBlockChatOutput) => void;
}) {
  const buttons: Array<{
    label: string;
    emptyBlock: ContentBlockChatOutput;
  }> = [
    {
      label: "Text",
      emptyBlock: {
        type: "text" as const,
        text: "",
      },
    },
    {
      label: "Tool Call",
      emptyBlock: {
        type: "tool_call" as const,
        id: "",
        name: null,
        raw_name: "",
        arguments: null,
        raw_arguments: "{}",
      },
    },
    {
      label: "Thought",
      emptyBlock: {
        type: "thought" as const,
        text: "",
      },
    },
  ];

  return (
    <div className="flex items-center gap-2 py-2">
      {buttons.map((button) => (
        <AddButton
          key={button.label}
          label={button.label}
          onAdd={() => onAdd(button.emptyBlock)}
        />
      ))}
      {/* TODO: we need to support adding other kinds of content blocks */}
      <span className="text-fg-muted text-xs">
        Please use the API or SDK for other content block types.
      </span>
    </div>
  );
}
