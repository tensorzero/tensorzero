import { useEffect, useState } from "react";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
  JsonValue,
} from "~/types/tensorzero";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  SnippetMessage,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { CodeEditor } from "../ui/code-editor";
import { ContentBlockRenderer } from "~/components/layout/ContentBlockRenderer";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";

/*
NOTE: This is the new output component but it is not editable yet so we are rolling
it out across the UI incrementally.
*/

export type ChatInferenceOutputRenderingData = ContentBlockChatOutput[];

export interface JsonInferenceOutputRenderingData extends JsonInferenceOutput {
  schema?: JsonValue;
}

// Base interface with just the common required properties
interface BaseOutputProps {
  output: ChatInferenceOutputRenderingData | JsonInferenceOutputRenderingData;
  maxHeight?: number;
}

// For when isEditing is not provided (default behavior)
interface DefaultOutputProps extends BaseOutputProps {
  isEditing?: never;
  onOutputChange?: never;
}

// For when isEditing is explicitly provided
interface EditableOutputProps extends BaseOutputProps {
  isEditing: boolean;
  onOutputChange: (
    output:
      | ChatInferenceOutputRenderingData
      | JsonInferenceOutputRenderingData
      | null,
  ) => void;
}

type OutputProps = DefaultOutputProps | EditableOutputProps;

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutputRenderingData {
  return "raw" in output;
}

interface JsonInferenceOutputComponentProps {
  output: JsonInferenceOutputRenderingData;
  maxHeight?: number;
  isEditing?: boolean;
  onOutputChange?: (output: JsonInferenceOutputRenderingData | null) => void;
}

function JsonInferenceOutputComponent({
  output,
  maxHeight,
  isEditing,
  onOutputChange,
}: JsonInferenceOutputComponentProps) {
  const [displayValue, setDisplayValue] = useState<string | undefined>(
    output.raw ?? undefined,
  );
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | undefined>();

  useEffect(() => {
    // Update display value when output.raw changes externally
    setDisplayValue(output.raw ?? undefined);
  }, [output.raw]);

  useEffect(() => {
    // Switch to raw tab when entering edit mode
    if (isEditing) {
      setActiveTab("raw");
    }
  }, [isEditing]);

  const handleRawChange = (value: string | undefined) => {
    if (onOutputChange && value !== undefined) {
      setDisplayValue(value);

      try {
        // Attempt to parse the JSON to validate it
        const parsedValue = JSON.parse(value);
        setJsonError(null);
        onOutputChange({
          ...output,
          raw: value,
          parsed: parsedValue,
        });
      } catch {
        setJsonError("Invalid JSON format");
        onOutputChange({
          ...output,
          raw: value,
          parsed: null,
        });
      }
    }
  };
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

  // Set default tab to Raw when editing, otherwise Parsed if it has content
  const defaultTab = isEditing ? "raw" : output.parsed ? "parsed" : "raw";

  return (
    <SnippetTabs
      tabs={tabs}
      defaultTab={defaultTab}
      activeTab={activeTab}
      onTabChange={setActiveTab}
    >
      {(currentTab) => (
        <SnippetContent maxHeight={maxHeight}>
          {currentTab === "parsed" ? (
            <>
              {output.parsed ? (
                <>
                  <CodeEditor
                    allowedLanguages={["json"]}
                    value={JSON.stringify(output.parsed, null, 2)}
                    readOnly
                  />
                  {isEditing && jsonError && (
                    <div className="mt-2 text-sm text-red-500">{jsonError}</div>
                  )}
                </>
              ) : (
                <EmptyMessage message="The inference output failed to parse against the schema." />
              )}
            </>
          ) : currentTab === "raw" ? (
            <>
              <CodeEditor
                allowedLanguages={["json"]}
                value={isEditing ? displayValue : (output.raw ?? undefined)}
                readOnly={!isEditing}
                onChange={isEditing ? handleRawChange : undefined}
              />
              {isEditing && jsonError && (
                <div className="mt-2 text-sm text-red-500">{jsonError}</div>
              )}
            </>
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

interface ChatInferenceOutputComponentProps {
  output: ChatInferenceOutputRenderingData;
  maxHeight?: number;
  isEditing?: boolean;
  onOutputChange?: (output: ChatInferenceOutputRenderingData) => void;
}

function ChatInferenceOutputComponent({
  output,
  maxHeight,
  isEditing,
  onOutputChange,
}: ChatInferenceOutputComponentProps) {
  const handleBlockChange = (
    index: number,
    updatedBlock: ContentBlockChatOutput,
  ) => {
    if (onOutputChange) {
      const updatedBlocks = [...output];
      updatedBlocks[index] = updatedBlock;
      onOutputChange(updatedBlocks);
    }
  };

  const onDeleteContentBlock = (index: number) => {
    if (onOutputChange) {
      const updatedBlocks = [...output];
      updatedBlocks.splice(index, 1);
      onOutputChange(updatedBlocks);
    }
  };

  const onAppendContentBlock = (contentBlock: ContentBlockChatOutput) => {
    if (onOutputChange) {
      const updatedBlocks = [...output, contentBlock];
      onOutputChange(updatedBlocks);
    }
  };

  const onAppendTextContentBlock = () => {
    const contentBlock: ContentBlockChatOutput = {
      type: "text",
      text: "",
    };
    onAppendContentBlock(contentBlock);
  };

  const onAppendToolCallContentBlock = () => {
    const contentBlock: ContentBlockChatOutput = {
      type: "tool_call",
      id: "",
      name: "",
      raw_name: "",
      arguments: null,
      raw_arguments: "{}",
    };
    onAppendContentBlock(contentBlock);
  };

  return (
    <SnippetContent maxHeight={maxHeight}>
      {output.length === 0 && !isEditing ? (
        <EmptyMessage message="The output was empty" />
      ) : (
        <SnippetMessage>
          {output.map((block, index) => (
            <ContentBlockRenderer
              key={String(index)}
              block={block}
              isEditing={isEditing}
              onChange={(updatedBlock) =>
                handleBlockChange(index, updatedBlock as ContentBlockChatOutput)
              }
              action={
                isEditing && (
                  <DeleteButton
                    onDelete={() => onDeleteContentBlock(index)}
                    label="Delete content block"
                  />
                )
              }
              thoughtFooter={(thoughtBlock) =>
                thoughtBlock.signature ? (
                  <>
                    Signature:{" "}
                    <span className="font-mono text-xs">
                      {thoughtBlock.signature}
                    </span>
                  </>
                ) : null
              }
            />
          ))}
          {isEditing && (
            <div className="flex items-center gap-2 py-2">
              <AddButton label="Text" onAdd={onAppendTextContentBlock} />
              <AddButton
                label="Tool Call"
                onAdd={onAppendToolCallContentBlock}
              />
            </div>
          )}
        </SnippetMessage>
      )}
    </SnippetContent>
  );
}

export function Output({
  output,
  maxHeight,
  isEditing,
  onOutputChange,
}: OutputProps) {
  return (
    <SnippetLayout>
      {isJsonInferenceOutput(output) ? (
        <JsonInferenceOutputComponent
          output={output}
          maxHeight={maxHeight}
          isEditing={isEditing}
          onOutputChange={onOutputChange}
        />
      ) : (
        <ChatInferenceOutputComponent
          output={output}
          maxHeight={maxHeight}
          isEditing={isEditing}
          onOutputChange={onOutputChange}
        />
      )}
    </SnippetLayout>
  );
}
