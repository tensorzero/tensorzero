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
import {
  CodeMessage,
  TextMessage,
  ToolCallMessage,
} from "~/components/layout/SnippetContent";

/*
NOTE: This is the new output component but it is not editable yet so we are rolling
it out across the UI incrementally.
*/

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
      return (
        <SnippetMessage key={index}>
          <TextMessage label="Text" content={block.text} />
        </SnippetMessage>
      );
    case "tool_call":
      return (
        <SnippetMessage key={index}>
          <ToolCallMessage
            toolName={block.name || "Tool call"}
            toolArguments={JSON.stringify(block.arguments || {}, null, 2)}
            toolCallId={block.id}
          />
        </SnippetMessage>
      );
  }
}

export default function Output({ output, outputSchema }: OutputProps) {
  return (
    <SnippetLayout>
      {isJsonInferenceOutput(output) ? (
        // JSON output with tabs
        (() => {
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
                    <SnippetMessage>
                      <CodeMessage
                        content={
                          output.parsed
                            ? JSON.stringify(output.parsed, null, 2)
                            : "The inference output failed to parse against the schema."
                        }
                        showLineNumbers={true}
                      />
                    </SnippetMessage>
                  ) : activeTab === "raw" ? (
                    <SnippetMessage>
                      <CodeMessage
                        content={output.raw}
                        showLineNumbers={true}
                      />
                    </SnippetMessage>
                  ) : (
                    <SnippetMessage>
                      <CodeMessage
                        content={JSON.stringify(outputSchema, null, 2)}
                        showLineNumbers={true}
                      />
                    </SnippetMessage>
                  )}
                </SnippetContent>
              )}
            </SnippetTabs>
          );
        })()
      ) : (
        // Content blocks output
        <SnippetContent>
          {output.map((block, index) => renderContentBlock(block, index))}
        </SnippetContent>
      )}
    </SnippetLayout>
  );
}
