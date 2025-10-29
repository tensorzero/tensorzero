import { Terminal } from "lucide-react";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { ToolPayload } from "~/components/input_output/content_blocks/ToolPayload";
import { type ToolCallInput } from "~/types/tensorzero";

interface ToolCallContentBlockProps {
  block: ToolCallInput; // TODO: `ToolCallInput` vs `ToolCall`? DO NOT MERGE
  isEditing?: boolean;
  onChange?: (updatedBlock: ToolCallInput) => void;
  actionBar?: ReactNode;
}

export function ToolCallContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: ToolCallContentBlockProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<Terminal className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Tool Call
      </ContentBlockLabel>
      <ToolPayload
        name={block.name || block.raw_name}
        id={block.id}
        payload={block.arguments}
        payloadLabel="Arguments"
        isEditing={isEditing}
        onChange={(id, name, payload) => {
          onChange?.({ ...block, id, name, arguments: payload });
        }}
        enforceJson={true}
      />
    </div>
  );
}
