import type { ZodModelInferenceOutputContentBlock } from "~/utils/clickhouse/common";
import { SnippetLayout } from "../layout/SnippetLayout";
import { EmptyMessage } from "../layout/SnippetContent";
import { TextMessage, ToolCallMessage } from "../layout/SnippetContent";

interface ModelInferenceOutputProps {
  output: ZodModelInferenceOutputContentBlock[];
}

export default function ModelInferenceOutput({
  output,
}: ModelInferenceOutputProps) {
  return (
    <SnippetLayout>
      {output.length === 0 ? (
        <EmptyMessage message="The output was empty" />
      ) : (
        output.map((block, index) => {
          switch (block.type) {
            case "text":
              return (
                <TextMessage key={index} label="Text" content={block.text} />
              );
            case "tool_call":
              return (
                <ToolCallMessage
                  key={index}
                  toolName={block.name}
                  toolArguments={block.arguments}
                  toolCallId={block.id}
                />
              );
            case "thought":
              return (
                <TextMessage
                  key={index}
                  label="Thought"
                  content={block.text || ""}
                  isEditing={false}
                  onChange={() => {}}
                />
              );
            case "unknown":
              // TODO: code editor should format as JSON by default
              return (
                <TextMessage
                  key={index}
                  label="Unknown Content"
                  content={JSON.stringify(block.data)}
                />
              );
          }
        })
      )}
    </SnippetLayout>
  );
}
