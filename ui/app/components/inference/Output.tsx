import { useEffect, useState } from "react";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
  JsonValue,
} from "tensorzero-node";
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
  return (
    <SnippetContent maxHeight={maxHeight}>
      {output.length === 0 ? (
        <EmptyMessage message="The output was empty" />
      ) : (
        <SnippetMessage>
          {output.map((block, index) => {
            switch (block.type) {
              case "text":
                return isEditing && onOutputChange ? (
                  <EditableTextMessage
                    key={index}
                    block={block}
                    onBlockChange={(updatedBlock) =>
                      handleBlockChange(index, updatedBlock)
                    }
                  />
                ) : (
                  <TextMessage key={index} label="Text" content={block.text} />
                );
              case "tool_call":
                return isEditing && onOutputChange ? (
                  <EditableToolCallMessage
                    key={index}
                    block={block}
                    onBlockChange={(updatedBlock) =>
                      handleBlockChange(index, updatedBlock)
                    }
                  />
                ) : (
                  <ToolCallMessage
                    key={index}
                    toolName={block.name ?? undefined}
                    toolRawName={block.raw_name}
                    toolArguments={
                      block.arguments
                        ? JSON.stringify(block.arguments, null, 2)
                        : undefined
                    }
                    toolRawArguments={block.raw_arguments}
                    toolCallId={block.id}
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
              case "thought": {
                const footer = block.signature ? (
                  <>
                    Signature:{" "}
                    <span className="font-mono text-xs">{block.signature}</span>
                  </>
                ) : null;

                return (
                  <TextMessage
                    key={index}
                    label="Thought"
                    content={block.text || ""}
                    footer={footer}
                  />
                );
              }
            }
          })}
        </SnippetMessage>
      )}
    </SnippetContent>
  );
}

// Editable Text Message component
interface EditableTextMessageProps {
  block: Extract<ContentBlockChatOutput, { type: "text" }>;
  onBlockChange: (
    updatedBlock: Extract<ContentBlockChatOutput, { type: "text" }>,
  ) => void;
}

function EditableTextMessage({
  block,
  onBlockChange,
}: EditableTextMessageProps) {
  const handleChange = (value: string | undefined) => {
    if (value !== undefined) {
      onBlockChange({
        ...block,
        text: value,
      });
    }
  };

  return (
    <div className="space-y-2">
      <div className="text-muted-foreground text-xs font-medium">Text</div>
      <CodeEditor
        allowedLanguages={["markdown"]}
        value={block.text}
        onChange={handleChange}
      />
    </div>
  );
}

// Editable Tool Call Message component
interface EditableToolCallMessageProps {
  block: Extract<ContentBlockChatOutput, { type: "tool_call" }>;
  onBlockChange: (
    updatedBlock: Extract<ContentBlockChatOutput, { type: "tool_call" }>,
  ) => void;
}

function EditableToolCallMessage({
  block,
  onBlockChange,
}: EditableToolCallMessageProps) {
  const [displayValue, setDisplayValue] = useState(
    JSON.stringify(block.arguments, null, 2),
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    // Update display value when block.arguments changes externally
    setDisplayValue(JSON.stringify(block.arguments, null, 2));
  }, [block.arguments]);

  const handleChange = (value: string | undefined) => {
    if (value !== undefined) {
      setDisplayValue(value);

      try {
        const parsedValue = JSON.parse(value);
        setJsonError(null);
        onBlockChange({
          ...block,
          arguments: parsedValue,
          raw_arguments: value,
        });
      } catch {
        setJsonError("Invalid JSON format");
      }
    }
  };

  return (
    <div className="space-y-2">
      <div className="text-muted-foreground text-xs font-medium">
        Tool: {block.name}
      </div>
      <CodeEditor
        allowedLanguages={["json"]}
        value={displayValue}
        onChange={handleChange}
      />
      {jsonError && <div className="text-sm text-red-500">{jsonError}</div>}
    </div>
  );
}

export function Output({
  output,
  maxHeight,
  isEditing,
  onOutputChange,
}: OutputProps) {
  const handleJsonOutputChange = (
    updatedOutput: JsonInferenceOutputRenderingData | null,
  ) => {
    if (onOutputChange) {
      onOutputChange(updatedOutput);
    }
  };

  const handleChatOutputChange = (
    updatedOutput: ChatInferenceOutputRenderingData,
  ) => {
    if (onOutputChange) {
      onOutputChange(updatedOutput);
    }
  };

  return (
    <SnippetLayout>
      {isJsonInferenceOutput(output) ? (
        <JsonInferenceOutputComponent
          output={output}
          maxHeight={maxHeight}
          isEditing={isEditing}
          onOutputChange={handleJsonOutputChange}
        />
      ) : (
        <ChatInferenceOutputComponent
          output={output}
          maxHeight={maxHeight}
          isEditing={isEditing}
          onOutputChange={handleChatOutputChange}
        />
      )}
    </SnippetLayout>
  );
}
