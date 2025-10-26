import { EmptyMessage } from "./ContentBlockElement";
import type {
  JsonValue,
  ResolvedInput,
  ResolvedInputMessage,
} from "~/types/tensorzero";
import SystemElement from "./SystemElement";
import MessagesElement from "./MessagesElement";

interface ResolvedInputElementProps {
  input: ResolvedInput;
  isEditing?: boolean;
  onSystemChange?: (system: JsonValue) => void;
  onMessagesChange?: (messages: ResolvedInputMessage[]) => void;
  maxHeight?: number | "Content";
}

export default function ResolvedInputElement({
  input,
  isEditing,
  onSystemChange,
  onMessagesChange,
  maxHeight,
}: ResolvedInputElementProps) {
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
