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
  EmptyMessage,
  TextMessage,
  ToolCallMessage,
} from "~/components/layout/SnippetContent";

/*
NOTE: This is the new output component but it is not editable yet so we are rolling
it out across the UI incrementally.
*/

type ChatInferenceOutputRenderingData = ContentBlockOutput[];

interface JsonInferenceOutputRenderingData extends JsonInferenceOutput {
  schema?: Record<string, unknown>;
}

interface OutputProps {
  output: ChatInferenceOutputRenderingData | JsonInferenceOutputRenderingData;
}

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutputRenderingData {
  return "raw" in output;
}

function renderJsonInferenceOutput(output: JsonInferenceOutputRenderingData) {
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
  if (output.schema) {
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
              {output.parsed ? (
                <CodeMessage
                  content={JSON.stringify(output.parsed, null, 2)}
                  showLineNumbers={true}
                />
              ) : (
                <EmptyMessage message="The inference output failed to parse against the schema." />
              )}
            </SnippetMessage>
          ) : activeTab === "raw" ? (
            <SnippetMessage>
              <CodeMessage content={output.raw} showLineNumbers={true} />
            </SnippetMessage>
          ) : (
            <SnippetMessage>
              <CodeMessage
                content={JSON.stringify(output.schema, null, 2)}
                showLineNumbers={true}
              />
            </SnippetMessage>
          )}
        </SnippetContent>
      )}
    </SnippetTabs>
  );
}

function renderChatInferenceOutput(output: ChatInferenceOutputRenderingData) {
  return (
    <SnippetContent>
      <SnippetMessage>
        {output.length === 0 ? (
          <EmptyMessage message="No output messages found" />
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
                    toolName={block.name ?? ""}
                    toolArguments={JSON.stringify(block.arguments, null, 2)}
                    // TODO: if arguments is null, display raw arguments without parsing
                    toolCallId={block.id}
                  />
                );
              default:
                return null;
            }
          })
        )}
      </SnippetMessage>
    </SnippetContent>
  );
}

export default function Output({ output }: OutputProps) {
  return (
    <SnippetLayout>
      {isJsonInferenceOutput(output)
        ? renderJsonInferenceOutput(output)
        : renderChatInferenceOutput(output)}
    </SnippetLayout>
  );
}
