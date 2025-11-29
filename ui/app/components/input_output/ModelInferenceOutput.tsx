import type { ZodModelInferenceOutputContentBlock } from "~/utils/clickhouse/common";
import { TextContentBlock } from "~/components/input_output/content_blocks/TextContentBlock";
import { ToolCallContentBlock } from "~/components/input_output/content_blocks/ToolCallContentBlock";
import { ThoughtContentBlock } from "~/components/input_output/content_blocks/ThoughtContentBlock";
import { EmptyMessage } from "~/components/input_output/ContentBlockElement";

interface ModelInferenceOutputProps {
  output: ZodModelInferenceOutputContentBlock[];
}

export default function ModelInferenceOutput({
  output,
}: ModelInferenceOutputProps) {
  if (output.length === 0) {
    return (
      <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
        <EmptyMessage message="The output was empty" />
      </div>
    );
  }

  return (
    <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
      {output.map((block, index) => {
        switch (block.type) {
          case "text":
            return (
              <TextContentBlock key={index} label="Text" text={block.text} />
            );
          case "tool_call":
            return <ToolCallContentBlock key={index} block={block} />;
          case "thought":
            return <ThoughtContentBlock key={index} block={block} />;
          case "unknown":
            return (
              <TextContentBlock
                key={index}
                label="Unknown"
                text={JSON.stringify(block.data)}
              />
            );
        }
      })}
    </div>
  );
}
