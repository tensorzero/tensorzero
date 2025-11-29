import type { InputMessage, InputMessageContent } from "~/types/tensorzero";
import { ContentBlockElement } from "./ContentBlockElement";
import { MessageWrapper } from "./MessageWrapper";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";

export function MessageElement({
  message,
  key,
  isEditing,
  onChange,
  onDelete,
}: {
  message: InputMessage;
  key: string;
  isEditing?: boolean;
  onChange?: (updatedMessage: InputMessage) => void;
  onDelete?: () => void;
}) {
  const onAppendContentBlock = (contentBlock: InputMessageContent) => {
    const updatedMessage = { ...message };
    updatedMessage.content = [...updatedMessage.content, contentBlock];
    onChange?.(updatedMessage);
  };

  const onUpdateContentBlock = (
    contentBlockIndex: number,
    updatedContentBlock: InputMessageContent,
  ) => {
    const updatedMessage = { ...message };
    const updatedContent = [...updatedMessage.content];
    updatedContent[contentBlockIndex] = updatedContentBlock;
    updatedMessage.content = updatedContent;
    onChange?.(updatedMessage);
  };

  const onDeleteContentBlock = (contentBlockIndex: number) => {
    const updatedMessage = { ...message };
    const updatedContent = [...updatedMessage.content];
    updatedContent.splice(contentBlockIndex, 1);
    updatedMessage.content = updatedContent;
    onChange?.(updatedMessage);
  };

  return (
    <MessageWrapper
      role={message.role}
      key={key}
      actionBar={
        isEditing && <DeleteButton onDelete={onDelete} label="Delete message" />
      }
    >
      {message.content.length === 0 && !isEditing && (
        <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
          Empty message
        </div>
      )}
      {message.content.map((block, contentBlockIndex) => (
        <ContentBlockElement
          key={`${key}-${contentBlockIndex}`}
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
  onAdd: (block: InputMessageContent) => void;
}) {
  const buttons: Array<{
    label: string;
    emptyBlock: InputMessageContent;
  }> = [
    {
      label: "Text",
      emptyBlock: {
        type: "text" as const,
        text: "",
      },
    },
    {
      label: "Template",
      emptyBlock: {
        type: "template" as const,
        name: "",
        arguments: {},
      },
    },
    {
      label: "Tool Call",
      emptyBlock: {
        type: "tool_call" as const,
        name: "",
        id: "",
        arguments: "{}",
      },
    },
    {
      label: "Tool Result",
      emptyBlock: {
        type: "tool_result" as const,
        name: "",
        id: "",
        result: "",
      },
    },
    {
      label: "Thought",
      emptyBlock: {
        type: "thought" as const,
        text: "",
      },
    },
    {
      label: "File (URL)",
      emptyBlock: {
        type: "file" as const,
        file_type: "url" as const,
        url: "",
        mime_type: null,
      },
    },
    {
      label: "File (Base64)",
      emptyBlock: {
        type: "file" as const,
        file_type: "base64" as const,
        mime_type: "",
        data: "",
      },
    },
    {
      label: "Unknown",
      emptyBlock: {
        type: "unknown" as const,
        data: {},
      },
    },
  ];

  return (
    <div className="flex flex-wrap items-center gap-2 py-2">
      {buttons.map((button) => (
        <AddButton
          key={button.label}
          label={button.label}
          onAdd={() => onAdd(button.emptyBlock)}
        />
      ))}
    </div>
  );
}
