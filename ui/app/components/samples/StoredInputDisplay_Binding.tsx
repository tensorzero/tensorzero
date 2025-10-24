import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import {
  TextMessage,
  EmptyMessage,
  TemplateMessage,
  ContentBlockDisplay,
} from "./StoredInputContentBlockDisplay_Binding";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";
import type {
  StoredInput,
  StoredInputMessage,
  StoredInputMessageContent,
  JsonValue,
  ToolCall,
  ToolResult,
  Thought,
} from "tensorzero-node";

interface InputDisplayProps {
  input: StoredInput;
  isEditing?: boolean;
  onSystemChange?: (system: JsonValue) => void;
  onMessagesChange?: (messages: StoredInputMessage[]) => void;
  maxHeight?: number | "Content";
}

export default function InputDisplay({
  input,
  isEditing,
  onSystemChange,
  onMessagesChange,
  maxHeight,
}: InputDisplayProps) {
  const { messages, system } = input;
  const onContentBlockChange = (
    messageIndex: number,
    contentBlockIndex: number,
    updatedContentBlock: StoredInputMessageContent,
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
    contentBlock: StoredInputMessageContent,
  ) => {
    const updatedMessage: StoredInputMessage = {
      role: messages[messageIndex].role,
      content: [...messages[messageIndex].content, contentBlock],
    };

    const updatedMessages = [...messages];
    updatedMessages[messageIndex] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  const onAppendTextContentBlock = (messageIndex: number) => {
    const contentBlock = {
      type: "text" as const,
      value: "",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendToolCallContentBlock = (messageIndex: number) => {
    const contentBlock: { type: "tool_call" } & ToolCall = {
      type: "tool_call" as const,
      id: "",
      name: "",
      arguments: "{}",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendToolResultContentBlock = (messageIndex: number) => {
    const contentBlock: { type: "tool_result" } & ToolResult = {
      type: "tool_result" as const,
      name: "",
      id: "",
      result: "",
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendTemplateContentBlock = (messageIndex: number) => {
    const contentBlock = {
      type: "template" as const,
      name: "",
      arguments: JSON.parse("{}"),
    };

    onAppendContentBlock(messageIndex, contentBlock);
  };

  const onAppendThoughtContentBlock = (messageIndex: number) => {
    const contentBlock: { type: "thought" } & Thought = {
      type: "thought" as const,
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
                <ContentBlockDisplay
                  key={`${messageIndex}-${contentBlockIndex}`}
                  block={block}
                  isEditing={isEditing}
                  onChange={(updatedContentBlock) =>
                    onContentBlockChange(
                      messageIndex,
                      contentBlockIndex,
                      updatedContentBlock,
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
  system?: JsonValue;
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
  }

  const systemDisplay =
    typeof system === "string" ? (
      <TextMessage
        content={system}
        isEditing={isEditing}
        onChange={onSystemChange}
      />
    ) : (
      <TemplateMessage
        templateArguments={system}
        templateName="system"
        isEditing={isEditing}
        onChange={(_templateName, value) => onSystemChange?.(value)}
      />
    );

  // In editing mode, show a delete button in the action bar
  const action = isEditing ? (
    <AddButton
      label="Template"
      onAdd={() => onSystemChange?.(JSON.parse("{}"))}
    />
  ) : undefined;

  return (
    <SnippetContent maxHeight={maxHeight}>
      <SnippetMessage role="system" action={action}>
        {systemDisplay}
      </SnippetMessage>
    </SnippetContent>
  );
}
