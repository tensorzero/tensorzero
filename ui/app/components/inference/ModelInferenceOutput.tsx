import type { ModelInferenceOutputContentBlock } from "~/utils/clickhouse/common";
import { SnippetLayout } from "../layout/SnippetLayout";
import { EmptyMessage } from "../layout/SnippetContent";
import { TextMessage, ToolCallMessage } from "../layout/SnippetContent";

interface ModelInferenceOutputProps {
  output: ModelInferenceOutputContentBlock[];
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
          }
        })
      )}
    </SnippetLayout>
  );
}
