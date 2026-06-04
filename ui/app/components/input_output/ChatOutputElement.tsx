import type { ContentBlockChatOutput } from "~/types/tensorzero";
import { EmptyMessage } from "./ContentBlockElement";
import { ExpandableElement } from "./ExpandableElement";
import { ChatOutputMessageElement } from "./ChatOutputMessageElement";
import { ScrollFadeContainer } from "./ScrollFadeContainer";
import { AddButton } from "~/components/ui/AddButton";
import { cn } from "~/utils/common";
import type { ContentOverflow } from "./content_overflow";

interface ChatOutputElementProps {
  output?: ContentBlockChatOutput[];
  isEditing?: boolean;
  onOutputChange?: (output?: ContentBlockChatOutput[]) => void;
  overflow?: ContentOverflow;
}

export function ChatOutputElement({
  output,
  isEditing,
  onOutputChange,
  overflow,
}: ChatOutputElementProps) {
  const onAddOutput = () => {
    onOutputChange?.([]);
  };

  const onDeleteOutput = () => {
    onOutputChange?.(undefined);
  };

  if (output === undefined) {
    return (
      <div
        className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4"
        data-testid="chat-output"
      >
        {isEditing ? (
          <AddOutputButtons onAdd={onAddOutput} />
        ) : (
          <EmptyMessage message="No output" />
        )}
      </div>
    );
  }

  const content = (
    <ChatOutputMessageElement
      output={output}
      isEditing={isEditing}
      onChange={onOutputChange}
      onDelete={onDeleteOutput}
    />
  );

  const isScroll = overflow?.type === "scroll";

  return (
    <div
      className={cn(
        "bg-bg-primary border-border flex w-full flex-col rounded-lg border",
        isScroll ? "flex-1 gap-0" : "gap-1 p-4",
      )}
      data-testid="chat-output"
    >
      {isScroll ? (
        <ScrollFadeContainer
          maxHeight={overflow.maxHeight}
          contentClassName="gap-1 px-4"
        >
          {content}
        </ScrollFadeContainer>
      ) : (
        <ExpandableElement maxHeight={overflow?.maxHeight}>
          {content}
        </ExpandableElement>
      )}
    </div>
  );
}

function AddOutputButtons({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex items-center gap-2 py-2">
      <AddButton label="Output" onAdd={onAdd} />
    </div>
  );
}
