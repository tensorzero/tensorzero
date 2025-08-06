import { useEffect, useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent } from "~/components/ui/card";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
} from "tensorzero-node";

// Base interface with just the common required properties
interface BaseOutputProps {
  output: JsonInferenceOutput | ContentBlockChatOutput[];
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
    output: JsonInferenceOutput | ContentBlockChatOutput[] | null,
  ) => void;
}

type OutputProps = DefaultOutputProps | EditableOutputProps;

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutput {
  return "raw" in output;
}

// ContentBlock Props
interface BaseContentBlockProps {
  block: ContentBlockChatOutput;
  index: number;
}

interface DefaultContentBlockProps extends BaseContentBlockProps {
  isEditing?: never;
  onBlockChange?: never;
}

interface EditableContentBlockProps extends BaseContentBlockProps {
  isEditing: boolean;
  onBlockChange: (updatedBlock: ContentBlockChatOutput) => void;
}

type ContentBlockProps = DefaultContentBlockProps | EditableContentBlockProps;

function renderContentBlock({
  block,
  index,
  isEditing,
  onBlockChange,
}: ContentBlockProps) {
  switch (block.type) {
    case "text":
      return (
        <TextBlock
          key={index}
          block={block}
          isEditing={isEditing ?? false}
          onBlockChange={onBlockChange}
        />
      );
    case "tool_call":
      return (
        <OutputToolCallBlock
          key={index}
          block={block}
          isEditing={isEditing ?? false}
          onBlockChange={onBlockChange}
        />
      );
    default:
      return null;
  }
}

// TextBlock component
interface TextBlockProps {
  block: Extract<ContentBlockChatOutput, { type: "text" }>;
  isEditing?: boolean;
  onBlockChange?: (
    updatedBlock: Extract<ContentBlockChatOutput, { type: "text" }>,
  ) => void;
}

function TextBlock({ block, isEditing, onBlockChange }: TextBlockProps) {
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onBlockChange) {
      onBlockChange({
        ...block,
        text: e.target.value,
      });
    }
  };

  if (isEditing) {
    return (
      <div className="bg-muted rounded-md p-4">
        <Badge className="mb-2">Text</Badge>
        <textarea
          className="w-full rounded border border-slate-300 bg-white p-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-800"
          value={block.text}
          onChange={handleChange}
          rows={3}
        />
      </div>
    );
  }

  return (
    <div className="bg-muted rounded-md p-4">
      <Badge className="mb-2">Text</Badge>
      <pre className="overflow-x-auto break-words whitespace-pre-wrap">
        <code className="text-sm">{block.text}</code>
      </pre>
    </div>
  );
}

// ToolCallBlock component
interface OutputToolCallBlockProps {
  block: Extract<ContentBlockChatOutput, { type: "tool_call" }>;
  isEditing?: boolean;
  onBlockChange?: (
    updatedBlock: Extract<ContentBlockChatOutput, { type: "tool_call" }>,
  ) => void;
}
function OutputToolCallBlock({
  block,
  isEditing,
  onBlockChange,
}: OutputToolCallBlockProps) {
  const [displayValue, setDisplayValue] = useState(
    JSON.stringify(block.arguments, null, 2),
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    // Update display value when block.arguments changes externally
    setDisplayValue(JSON.stringify(block.arguments, null, 2));
  }, [block.arguments]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onBlockChange) {
      const newValue = e.target.value;
      setDisplayValue(newValue);

      try {
        const parsedValue = JSON.parse(newValue);
        setJsonError(null);
        onBlockChange({
          ...block,
          arguments: parsedValue,
          raw_arguments: newValue,
        });
      } catch {
        setJsonError("Invalid JSON format");
      }
    }
  };

  if (isEditing) {
    return (
      <div className="bg-muted rounded-md p-4">
        <Badge className="mb-2">Tool: {block.name}</Badge>
        <textarea
          className={`w-full rounded border bg-white p-2 font-mono text-sm ${
            jsonError
              ? "border-red-500 dark:border-red-500"
              : "border-slate-300 dark:border-slate-700"
          } dark:bg-slate-800`}
          value={displayValue}
          onChange={handleChange}
          rows={3}
        />
        {jsonError && (
          <div className="mt-1 text-sm text-red-500">{jsonError}</div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-muted rounded-md p-4">
      <Badge className="mb-2">Tool: {block.name}</Badge>
      <pre className="overflow-x-auto break-words whitespace-pre-wrap">
        <code className="text-sm">
          {JSON.stringify(block.arguments, null, 2)}
        </code>
      </pre>
    </div>
  );
}

// JsonOutput component for handling JsonInferenceOutput
interface JsonOutputProps {
  output: JsonInferenceOutput;
  isEditing?: boolean;
  onOutputChange?: (output: JsonInferenceOutput | null) => void; // null is used if the output is not valid JSON
}

function JsonOutput({ output, isEditing, onOutputChange }: JsonOutputProps) {
  const [displayValue, setDisplayValue] = useState(output.raw);
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    // Update display value when output.raw changes externally
    setDisplayValue(output.raw);
  }, [output.raw]);

  const handleRawChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onOutputChange) {
      const newValue = e.target.value;
      setDisplayValue(newValue);

      try {
        // Attempt to parse the JSON to validate it
        const parsedValue = JSON.parse(newValue);
        setJsonError(null);
        onOutputChange({
          ...output,
          raw: newValue,
          parsed: parsedValue,
        });
      } catch {
        setJsonError("Invalid JSON format");
        onOutputChange(null);
      }
    }
  };

  return (
    <div className="space-y-4">
      {output.parsed && (
        <div className="bg-muted rounded-md p-4">
          <Badge className="mb-2">Parsed Output</Badge>
          <pre className="overflow-x-auto break-words whitespace-pre-wrap">
            <code className="text-sm">
              {JSON.stringify(output.parsed, null, 2)}
            </code>
          </pre>
        </div>
      )}
      <div className="bg-muted rounded-md p-4">
        <Badge className="mb-2">Raw Output</Badge>
        {isEditing ? (
          <div>
            <textarea
              className={`w-full rounded border bg-white p-2 font-mono text-sm ${
                jsonError
                  ? "border-red-500 dark:border-red-500"
                  : "border-slate-300 dark:border-slate-700"
              } dark:bg-slate-800`}
              value={displayValue ?? undefined}
              onChange={handleRawChange}
              rows={5}
            />
            {jsonError && (
              <div className="mt-1 text-sm text-red-500">{jsonError}</div>
            )}
          </div>
        ) : (
          <pre className="overflow-x-auto break-words whitespace-pre-wrap">
            <code className="text-sm">{output.raw}</code>
          </pre>
        )}
      </div>
    </div>
  );
}

// ContentBlocksOutput component for handling ContentBlockOutput[]
interface ContentBlocksOutputProps {
  blocks: ContentBlockChatOutput[];
  isEditing?: boolean;
  onBlocksChange?: (blocks: ContentBlockChatOutput[]) => void;
}

function ContentBlocksOutput({
  blocks,
  isEditing,
  onBlocksChange,
}: ContentBlocksOutputProps) {
  const handleBlockChange = (
    index: number,
    updatedBlock: ContentBlockChatOutput,
  ) => {
    if (onBlocksChange) {
      const updatedBlocks = [...blocks];
      updatedBlocks[index] = updatedBlock;
      onBlocksChange(updatedBlocks);
    }
  };

  return (
    <div className="space-y-2">
      {blocks.map((block, index) =>
        renderContentBlock({
          block,
          index,
          isEditing: isEditing ?? false,
          onBlockChange: (updatedBlock) =>
            handleBlockChange(index, updatedBlock),
        }),
      )}
    </div>
  );
}

export function OutputContent({
  output,
  isEditing,
  onOutputChange,
}: OutputProps) {
  const handleJsonOutputChange = (
    updatedOutput: JsonInferenceOutput | null,
  ) => {
    if (onOutputChange) {
      onOutputChange(updatedOutput);
    }
  };

  const handleBlocksChange = (updatedBlocks: ContentBlockChatOutput[]) => {
    if (onOutputChange) {
      onOutputChange(updatedBlocks);
    }
  };

  return isJsonInferenceOutput(output) ? (
    <JsonOutput
      output={output}
      isEditing={isEditing ?? false}
      onOutputChange={handleJsonOutputChange}
    />
  ) : (
    <ContentBlocksOutput
      blocks={output}
      isEditing={isEditing ?? false}
      onBlocksChange={handleBlocksChange}
    />
  );
}

export default function Output({
  output,
  isEditing,
  onOutputChange,
}: OutputProps) {
  // Don't pass undefined values to OutputContent when in editing mode
  if (isEditing) {
    return (
      <Card>
        <CardContent className="pt-6">
          <OutputContent
            output={output}
            isEditing={isEditing}
            onOutputChange={onOutputChange}
          />
        </CardContent>
      </Card>
    );
  }

  // Default non-editing mode
  return (
    <Card>
      <CardContent className="pt-6">
        <OutputContent output={output} />
      </CardContent>
    </Card>
  );
}
