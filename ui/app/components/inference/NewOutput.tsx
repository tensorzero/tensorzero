import type { JsonInferenceOutput } from "tensorzero-node";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  SnippetMessage,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import {
  EmptyMessage,
  TextMessage,
  ToolCallMessage,
} from "~/components/layout/SnippetContent";
import { CodeEditor } from "../ui/code-editor";
import type { JsonValue, ContentBlockChatOutput } from "tensorzero-node";

// IMPORTANT: THIS VERSION HAS `maxHeight` IN THE PROPS WHICH THE OTHER VERSIONS DO NOT HAVE

/*
  NOTE: This is the new output component but it is not editable yet so we are rolling
  it out across the UI incrementally.
*/

export type ChatInferenceOutputRenderingData = ContentBlockChatOutput[];

export interface JsonInferenceOutputRenderingData extends JsonInferenceOutput {
  schema?: JsonValue;
}

interface OutputProps {
  output: ChatInferenceOutputRenderingData | JsonInferenceOutputRenderingData;
  maxHeight?: number;
}

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutputRenderingData {
  return "raw" in output;
}

function renderJsonInferenceOutput(
  output: JsonInferenceOutputRenderingData,
  maxHeight?: number,
) {
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

  // Set default tab to "Parsed" if it's available, otherwise "Raw"
  const defaultTab = output.parsed ? "parsed" : "raw";

  return (
    <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
      {(activeTab) => (
        <SnippetContent maxHeight={maxHeight}>
          {activeTab === "parsed" ? (
            <>
              {output.parsed ? (
                <CodeEditor
                  allowedLanguages={["json"]}
                  value={JSON.stringify(output.parsed, null, 2)}
                  readOnly
                />
              ) : (
                <EmptyMessage message="The inference output failed to parse against the schema." />
              )}
            </>
          ) : activeTab === "raw" ? (
            <CodeEditor
              allowedLanguages={["json"]}
              value={output.raw ?? undefined}
              readOnly
            />
          ) : (
            <CodeEditor
              allowedLanguages={["json"]}
              value={JSON.stringify(output.schema, null, 2)}
              readOnly
            />
          )}
        </SnippetContent>
      )}
    </SnippetTabs>
  );
}

function renderChatInferenceOutput(
  output: ChatInferenceOutputRenderingData,
  maxHeight?: number,
) {
  return (
    <SnippetContent maxHeight={maxHeight}>
      {output.length === 0 ? (
        <EmptyMessage message="The output was empty" />
      ) : (
        <SnippetMessage>
          {output.map((block, index) => {
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
                    toolRawName={block.raw_name}
                    toolArguments={JSON.stringify(block.arguments, null, 2)}
                    toolRawArguments={block.raw_arguments}
                    toolCallId={block.id}
                  />
                );
              default:
                return null;
            }
          })}
        </SnippetMessage>
      )}
    </SnippetContent>
  );
}

export default function Output({ output, maxHeight }: OutputProps) {
  return (
    <SnippetLayout>
      {isJsonInferenceOutput(output)
        ? renderJsonInferenceOutput(output, maxHeight)
        : renderChatInferenceOutput(output, maxHeight)}
    </SnippetLayout>
  );
}
