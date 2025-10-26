import { ArrowRight } from "lucide-react";
import { type ReactNode } from "react";
import ContentBlockLabel from "~/components/input_output/content_blocks/ContentBlockLabel";
import ToolPayload from "~/components/input_output/content_blocks/ToolPayload";
import { type ToolResult } from "~/types/tensorzero";

interface ToolResultContentBlockProps {
  block: ToolResult;
  isEditing?: boolean;
  onChange?: (updatedBlock: ToolResult) => void;
  actionBar?: ReactNode;
}

export default function ToolResultContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: ToolResultContentBlockProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<ArrowRight className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Tool Result
      </ContentBlockLabel>
      <ToolPayload
        name={block.name}
        id={block.id}
        payload={block.result}
        payloadLabel="Result"
        isEditing={isEditing}
        onChange={(id, name, payload) => {
          onChange?.({ ...block, id, name, result: payload });
        }}
      />
    </div>
  );
}
