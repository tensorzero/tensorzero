import type {
  DisplayInputMessageContent,
  DisplayInputMessage,
  DisplayTextInput,
  ToolCallContent,
  ToolResultContent,
  TemplateInput,
  ThoughtContent,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import {
  TextMessage,
  EmptyMessage,
  TemplateMessage,
} from "~/components/layout/SnippetContent";
import { ContentBlockRenderer } from "~/components/layout/ContentBlockRenderer";
import type { JsonObject } from "type-fest";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";

/// NOTE: This component is being DEPRECATED in favor of ui/app/components/input_output/ResolvedInputElement.tsx.

interface InputProps {
  messages: DisplayInputMessage[];
  system?: string | JsonObject | null;
  isEditing?: boolean;
  onSystemChange?: (system: string | object | null) => void;
  onMessagesChange?: (messages: DisplayInputMessage[]) => void;
  maxHeight?: number | "Content";
}

export default function Input({
  system,
  messages,
  isEditing,
  onSystemChange,
  onMessagesChange,
  maxHeight,
}: InputProps) {
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

  const onDeleteMessage = (messageIndex: number) => {
    const updatedMessages = [...messages];
    updatedMessages.splice(messageIndex, 1);
    onMessagesChange?.(updatedMessages);
  };

  const onDeleteContentBlock = (
    messageIndex: number,
    contentBlockIndex: number,
  ) => {
    const updatedMessages = [...messages];
    const updatedMessage = { ...updatedMessages[messageIndex] };
    const updatedContent = [...updatedMessage.content];
    updatedContent.splice(contentBlockIndex, 1);
    updatedMessage.content = updatedContent;
    updatedMessages[messageIndex] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  const onAppendMessage = (role: "user" | "assistant") => {
    const newMessage = {
      role,
      content: [],
    };
    const updatedMessages = [...messages, newMessage];
    onMessagesChange?.(updatedMessages);
  };

  const onAppendContentBlock = (
    messageIndex: number,
    contentBlock: DisplayInputMessageContent,
  ) => {
    const updatedMessage: DisplayInputMessage = {
      role: messages[messageIndex].role,
      content: [...messages[messageIndex].content, contentBlock],
    };

    const updatedMessages = [...messages];
    updatedMessages[messageIndex] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  const onAppendTextContentBlock = (messageIndex: number) => {
    const contentBlock: DisplayTextInput = {
      type: "text",
      text: "",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendToolCallContentBlock = (messageIndex: number) => {
    const contentBlock: ToolCallContent = {
      type: "tool_call",
      name: "",
      id: "",
      arguments: "{}",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendToolResultContentBlock = (messageIndex: number) => {
    const contentBlock: ToolResultContent = {
      type: "tool_result",
      name: "",
      id: "",
      result: "",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendTemplateContentBlock = (messageIndex: number) => {
    const contentBlock: TemplateInput = {
      type: "template",
      name: "",
      arguments: JSON.parse("{}"),
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendThoughtContentBlock = (messageIndex: number) => {
    const contentBlock: ThoughtContent = {
      type: "thought",
      text: "",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  return (
    <SnippetLayout>
      {/* Empty input */}
      {system == null && messages.length === 0 && !isEditing && (
        <SnippetContent maxHeight={maxHeight}>
          <EmptyMessage message="Empty input" />
        </SnippetContent>
      )}
      {/* System */}
      <SystemSnippet
        system={system}
        isEditing={isEditing}
        onSystemChange={onSystemChange}
        maxHeight={maxHeight}
      />
      {/* Messages */}
      {(messages.length > 0 || isEditing) && (
        <SnippetContent maxHeight={maxHeight}>
          {messages.map((message, messageIndex) => (
            <SnippetMessage
              role={message.role}
              key={messageIndex}
              action={
                isEditing && (
                  <DeleteButton
                    onDelete={() => onDeleteMessage?.(messageIndex)}
                    label="Delete message"
                  />
                )
              }
            >
              {message.content.length === 0 && !isEditing && (
                <EmptyMessage message="Empty message" />
              )}
              {message.content.map((block, contentBlockIndex) => (
                <ContentBlockRenderer
                  key={`${messageIndex}-${contentBlockIndex}`}
                  block={block}
                  isEditing={isEditing}
                  onChange={(updatedContentBlock) =>
                    onContentBlockChange(
                      messageIndex,
                      contentBlockIndex,
                      updatedContentBlock as DisplayInputMessageContent,
                    )
                  }
                  action={
                    isEditing && (
                      <DeleteButton
                        onDelete={() =>
                          onDeleteContentBlock?.(
                            messageIndex,
                            contentBlockIndex,
                          )
                        }
                        label="Delete content block"
                      />
                    )
                  }
                />
              ))}
              {isEditing && (
                <div className="flex items-center gap-2 py-2">
                  <AddButton
                    label="Text"
                    onAdd={() => onAppendTextContentBlock?.(messageIndex)}
                  />
                  <AddButton
                    label="Template"
                    onAdd={() => onAppendTemplateContentBlock?.(messageIndex)}
                  />
                  <AddButton
                    label="Tool Call"
                    onAdd={() => onAppendToolCallContentBlock?.(messageIndex)}
                  />
                  <AddButton
                    label="Tool Result"
                    onAdd={() => onAppendToolResultContentBlock?.(messageIndex)}
                  />
                  <AddButton
                    label="Thought"
                    onAdd={() => onAppendThoughtContentBlock?.(messageIndex)}
                  />
                  {/* TODO: we need to support adding other kinds of content blocks */}
                  <span className="text-fg-muted text-xs">
                    Please use the API or SDK for other content block types.
                  </span>
                </div>
              )}
            </SnippetMessage>
          ))}
          {isEditing && (
            <div className="flex items-center gap-2 py-2">
              <AddButton
                label="User Message"
                onAdd={() => onAppendMessage?.("user")}
              />

              <AddButton
                label="Assistant Message"
                onAdd={() => onAppendMessage?.("assistant")}
              />
            </div>
          )}
        </SnippetContent>
      )}
    </SnippetLayout>
  );
}

interface SystemSnippetProps {
  system?: string | JsonObject | null;
  isEditing?: boolean;
  onSystemChange?: (system: string | object | null) => void;
  maxHeight?: number | "Content";
}

function SystemSnippet({
  system,
  isEditing,
  onSystemChange,
  maxHeight,
}: SystemSnippetProps) {
  if (system == null) {
    return (
      isEditing && (
        <SnippetContent maxHeight={maxHeight}>
          <SnippetMessage role="system">
            <div className="flex items-center gap-2 py-2">
              <AddButton label="Text" onAdd={() => onSystemChange?.("")} />
              {/* TODO: we should hide the following button if this function has no variants with a `system` template; it'll error on submission */}
              <AddButton
                label="Template"
                onAdd={() => onSystemChange?.(JSON.parse("{}"))}
              />
            </div>
          </SnippetMessage>
        </SnippetContent>
      )
    );
  } else {
    return (
      <SnippetContent maxHeight={maxHeight}>
        <SnippetMessage
          role="system"
          action={
            isEditing && (
              <DeleteButton
                label="Delete system"
                onDelete={() => onSystemChange?.(null)}
              />
            )
          }
        >
          {typeof system === "object" ? (
            <TemplateMessage
              templateArguments={system}
              templateName="system"
              isEditing={isEditing}
              onChange={(_templateName, value) => onSystemChange?.(value)}
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
    );
  }
}
