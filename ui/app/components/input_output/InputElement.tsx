import { EmptyMessage } from "./ContentBlockElement";
import type { System, Input, InputMessage } from "~/types/tensorzero";
import { SystemElement } from "./SystemElement";
import { MessagesElement } from "./MessagesElement";
import { ScrollFadeContainer } from "./ScrollFadeContainer";
import { cn } from "~/utils/common";
import type { ContentOverflow } from "./content_overflow";

interface InputElementProps {
  input: Input;
  isEditing?: boolean;
  onSystemChange?: (system?: System) => void;
  onMessagesChange?: (messages: InputMessage[]) => void;
  overflow?: ContentOverflow;
}

export function InputElement({
  input,
  isEditing,
  onSystemChange,
  onMessagesChange,
  overflow,
}: InputElementProps) {
  const { messages, system } = input;

  // In scroll mode, disable per-section ExpandableElements so the whole card scrolls
  const childMaxHeight =
    overflow?.type === "scroll" ? ("Content" as const) : overflow?.maxHeight;

  const content = (
    <>
      {system == null && messages.length === 0 && !isEditing && (
        <EmptyMessage message="Empty input" />
      )}
      <SystemElement
        system={system}
        isEditing={isEditing}
        onSystemChange={onSystemChange}
        maxHeight={childMaxHeight}
      />
      <MessagesElement
        messages={messages}
        isEditing={isEditing}
        onMessagesChange={onMessagesChange}
        maxHeight={childMaxHeight}
      />
    </>
  );

  const isScroll = overflow?.type === "scroll";

  return (
    <div
      className={cn(
        "bg-bg-primary border-border flex w-full flex-col rounded-lg border",
        isScroll ? "flex-1 gap-0" : "gap-1 p-4",
      )}
    >
      {isScroll ? (
        <ScrollFadeContainer
          maxHeight={overflow.maxHeight}
          contentClassName="gap-1 px-4"
        >
          {content}
        </ScrollFadeContainer>
      ) : (
        content
      )}
    </div>
  );
}
