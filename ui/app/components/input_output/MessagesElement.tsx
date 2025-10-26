import type { ResolvedInputMessage } from "~/types/tensorzero";
import { AddButton } from "~/components/ui/AddButton";
import { ExpandableElement } from "./ExpandableElement";
import { MessageElement } from "./MessageElement";

interface MessagesElementProps {
  messages: Array<ResolvedInputMessage>;
  isEditing?: boolean;
  onMessagesChange?: (messages: ResolvedInputMessage[]) => void;
  maxHeight?: number | "Content";
}

export function MessagesElement({
  messages,
  isEditing,
  onMessagesChange,
  maxHeight,
}: MessagesElementProps) {
  const onAppendMessage = (role: "user" | "assistant") => {
    const newMessage = {
      role,
      content: [],
    };
    const updatedMessages = [...messages, newMessage];
    onMessagesChange?.(updatedMessages);
  };

  const onUpdateMessage = (
    messageIndex: number,
    updatedMessage: ResolvedInputMessage,
  ) => {
    const updatedMessages = [...messages];
    updatedMessages[messageIndex] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  const onDeleteMessage = (messageIndex: number) => {
    const updatedMessages = [...messages];
    updatedMessages.splice(messageIndex, 1);
    onMessagesChange?.(updatedMessages);
  };

  if (!messages.length) {
    if (isEditing) {
      return <AddMessageButtons onAdd={onAppendMessage} />;
    } else {
      return null;
    }
  }

  const messagesElements = messages.map((message, messageIndex) => (
    <MessageElement
      message={message}
      key={`message-${messageIndex}`}
      isEditing={isEditing}
      onChange={(updatedMessage) =>
        onUpdateMessage(messageIndex, updatedMessage)
      }
      onDelete={() => onDeleteMessage(messageIndex)}
    />
  ));

  return (
    <ExpandableElement maxHeight={maxHeight}>
      {messagesElements}
      {isEditing && <AddMessageButtons onAdd={onAppendMessage} />}
    </ExpandableElement>
  );
}

function AddMessageButtons({
  onAdd: onAdd,
}: {
  onAdd: (role: "user" | "assistant") => void;
}) {
  return (
    <div className="flex items-center gap-2 py-2">
      <AddButton label="User Message" onAdd={() => onAdd("user")} />
      <AddButton label="Assistant Message" onAdd={() => onAdd("assistant")} />
    </div>
  );
}
