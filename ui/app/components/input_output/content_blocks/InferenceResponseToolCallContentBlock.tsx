import { Terminal } from "lucide-react";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { ToolPayload } from "~/components/input_output/content_blocks/ToolPayload";
import type { InferenceResponseToolCall } from "~/types/tensorzero";

interface InferenceResponseToolCallContentBlockProps {
  block: InferenceResponseToolCall;
  isEditing?: boolean;
  onChange?: (updatedBlock: InferenceResponseToolCall) => void;
  actionBar?: ReactNode;
}

export function InferenceResponseToolCallContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: InferenceResponseToolCallContentBlockProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<Terminal className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Tool Call
      </ContentBlockLabel>
      <ToolPayload
        name={block.raw_name}
        id={block.id}
        payload={block.raw_arguments}
        payloadLabel="Arguments"
        isEditing={isEditing}
        onChange={(id, raw_name, raw_arguments) => {
          // On change, set parsed values to `null`; the API will discard and re-compute them.
          onChange?.({
            ...block,
            id,
            raw_name,
            raw_arguments,
            name: null,
            arguments: null,
          });
        }}
        enforceJson={true}
      />
    </div>
  );
}
