import { Terminal } from "lucide-react";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { ToolPayload } from "~/components/input_output/content_blocks/ToolPayload";
import type { ToolCallWrapper } from "~/types/tensorzero";

interface ToolCallContentBlockProps {
  block: ToolCallWrapper;
  isEditing?: boolean;
  onChange?: (updatedBlock: ToolCallWrapper) => void;
  actionBar?: ReactNode;
}

export function ToolCallContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: ToolCallContentBlockProps) {
  const name: string = "raw_name" in block ? block.raw_name : block.name;
  const payload: string =
    "raw_arguments" in block ? block.raw_arguments : block.arguments;

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<Terminal className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Tool Call
      </ContentBlockLabel>
      <ToolPayload
        name={name}
        id={block.id}
        payload={payload}
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
