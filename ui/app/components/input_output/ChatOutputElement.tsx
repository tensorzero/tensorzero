import type { ContentBlockChatOutput } from "~/types/tensorzero";
import { EmptyMessage } from "./ContentBlockElement";
import { ExpandableElement } from "./ExpandableElement";
import { ChatOutputMessageElement } from "./ChatOutputMessageElement";
import { AddButton } from "~/components/ui/AddButton";

interface ChatOutputElementProps {
  output?: ContentBlockChatOutput[];
  isEditing?: boolean;
  onOutputChange?: (output?: ContentBlockChatOutput[]) => void;
  maxHeight?: number | "Content";
}

export function ChatOutputElement({
  output,
  isEditing,
  onOutputChange,
  maxHeight,
}: ChatOutputElementProps) {
  const onAddOutput = () => {
    onOutputChange?.([]);
  };

  const onDeleteOutput = () => {
    onOutputChange?.(undefined);
  };

  if (output === undefined) {
    return (
      <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
        {isEditing ? (
          <AddOutputButtons onAdd={onAddOutput} />
        ) : (
          <EmptyMessage message="No output" />
        )}
      </div>
    );
  }

  return (
    <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
      <ExpandableElement maxHeight={maxHeight}>
        <ChatOutputMessageElement
          output={output}
          isEditing={isEditing}
          onChange={onOutputChange}
          onDelete={onDeleteOutput}
        />
      </ExpandableElement>
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
