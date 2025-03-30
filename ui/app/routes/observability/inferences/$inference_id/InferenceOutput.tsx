import type {
  JsonInferenceOutput,
  ContentBlockOutput,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  SnippetMessage,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { CodeMessage, TextMessage } from "~/components/layout/SnippetContent";

interface OutputProps {
  output: JsonInferenceOutput | ContentBlockOutput[];
  outputSchema?: Record<string, unknown>;
}

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutput {
  return "raw" in output;
}

function renderContentBlock(block: ContentBlockOutput, index: number) {
  switch (block.type) {
    case "text":
      return <TextMessage key={index} label="Text" content={block.text} />;
    case "tool_call":
      return (
        <TextMessage
          key={index}
          label={`Tool: ${block.name}`}
          content={JSON.stringify(block.arguments, null, 2)}
        />
      );
  }
}

export function OutputContent({ output, outputSchema }: OutputProps) {
  if (isJsonInferenceOutput(output)) {
    const tabs: SnippetTab[] = [
      {
        id: "parsed",
        label: "Parsed Output",
        indicator: output.parsed ? "content" : "fail",
      },
      {
        id: "raw",
        label: "Raw Output",
      },
    ];

    // Add Output Schema tab if available
    if (outputSchema) {
      tabs.push({
        id: "schema",
        label: "Output Schema",
      });
    }

    // Set default tab to Parsed if it has content, otherwise Raw
    const defaultTab = output.parsed ? "parsed" : "raw";

    return (
      <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
        {(activeTab) => (
          <SnippetContent>
            {activeTab === "parsed" ? (
              <CodeMessage
                content={
                  output.parsed
                    ? JSON.stringify(output.parsed, null, 2)
                    : "The inference output failed to parse against the schema."
                }
                showLineNumbers={true}
              />
            ) : activeTab === "raw" ? (
              <CodeMessage content={output.raw} showLineNumbers={true} />
            ) : (
              <CodeMessage
                content={JSON.stringify(outputSchema, null, 2)}
                showLineNumbers={true}
              />
            )}
          </SnippetContent>
        )}
      </SnippetTabs>
    );
  }

  return (
    <SnippetContent>
      <SnippetMessage>
        {output.map((block, index) => renderContentBlock(block, index))}
      </SnippetMessage>
    </SnippetContent>
  );
}

export default function Output({ output, outputSchema }: OutputProps) {
  return (
    <SnippetLayout className="w-full">
      <OutputContent output={output} outputSchema={outputSchema} />
    </SnippetLayout>
  );
}
