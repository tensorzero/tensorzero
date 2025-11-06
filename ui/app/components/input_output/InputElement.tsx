import { EmptyMessage } from "./ContentBlockElement";
import type { System, Input, InputMessage } from "~/types/tensorzero";
import { SystemElement } from "./SystemElement";
import { MessagesElement } from "./MessagesElement";

interface InputElementProps {
  input: Input;
  isEditing?: boolean;
  onSystemChange?: (system?: System) => void;
  onMessagesChange?: (messages: InputMessage[]) => void;
  maxHeight?: number | "Content";
}

export function InputElement({
  input,
  isEditing,
  onSystemChange,
  onMessagesChange,
  maxHeight,
}: InputElementProps) {
  const { messages, system } = input;

  return (
    <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
      {/* Empty input */}
      {system == null && messages.length === 0 && !isEditing && (
        <EmptyMessage message="Empty input" />
      )}
      {/* System */}
      <SystemElement
        system={system}
        isEditing={isEditing}
        onSystemChange={onSystemChange}
        maxHeight={maxHeight}
      />
      {/* Messages */}
      <MessagesElement
        messages={messages}
        isEditing={isEditing}
        onMessagesChange={onMessagesChange}
        maxHeight={maxHeight}
      />
    </div>
  );
}
