import { Badge } from "~/components/ui/badge";
import { Card, CardContent } from "~/components/ui/card";
import type {
  JsonInferenceOutput,
  ContentBlockOutput,
} from "~/utils/clickhouse/common";

// Base interface with just the common required properties
interface BaseOutputProps {
  output: JsonInferenceOutput | ContentBlockOutput[];
}

// For when isEditing is not provided (default behavior)
interface DefaultOutputProps extends BaseOutputProps {
  isEditing?: never;
  onOutputChange?: never;
}

// For when isEditing is explicitly provided
interface EditableOutputProps extends BaseOutputProps {
  isEditing: boolean;
  onOutputChange: (output: JsonInferenceOutput | ContentBlockOutput[]) => void;
}

type OutputProps = DefaultOutputProps | EditableOutputProps;

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutput {
  return "raw" in output;
}

// ContentBlock Props
interface BaseContentBlockProps {
  block: ContentBlockOutput;
  index: number;
}

interface DefaultContentBlockProps extends BaseContentBlockProps {
  isEditing?: never;
  onBlockChange?: never;
}

interface EditableContentBlockProps extends BaseContentBlockProps {
  isEditing: boolean;
  onBlockChange: (updatedBlock: ContentBlockOutput) => void;
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
        <ToolCallBlock
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
  block: Extract<ContentBlockOutput, { type: "text" }>;
  isEditing?: boolean;
  onBlockChange?: (
    updatedBlock: Extract<ContentBlockOutput, { type: "text" }>,
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
      <div className="rounded-md bg-muted p-4">
        <Badge className="mb-2">Text</Badge>
        <textarea
          className="w-full rounded border border-slate-300 p-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-800"
          value={block.text}
          onChange={handleChange}
          rows={3}
        />
      </div>
    );
  }

  return (
    <div className="rounded-md bg-muted p-4">
      <Badge className="mb-2">Text</Badge>
      <pre className="overflow-x-auto whitespace-pre-wrap break-words">
        <code className="text-sm">{block.text}</code>
      </pre>
    </div>
  );
}

// ToolCallBlock component
interface ToolCallBlockProps {
  block: Extract<ContentBlockOutput, { type: "tool_call" }>;
  isEditing?: boolean;
  onBlockChange?: (
    updatedBlock: Extract<ContentBlockOutput, { type: "tool_call" }>,
  ) => void;
}

function ToolCallBlock({
  block,
  isEditing,
  onBlockChange,
}: ToolCallBlockProps) {
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onBlockChange) {
      onBlockChange({
        ...block,
        arguments: JSON.parse(e.target.value),
        raw_arguments: e.target.value,
      });
    }
  };

  if (isEditing) {
    return (
      <div className="rounded-md bg-muted p-4">
        <Badge className="mb-2">Tool: {block.name}</Badge>
        <textarea
          className="w-full rounded border border-slate-300 p-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-800"
          value={JSON.stringify(block.arguments, null, 2)}
          onChange={handleChange}
          rows={3}
        />
      </div>
    );
  }

  return (
    <div className="rounded-md bg-muted p-4">
      <Badge className="mb-2">Tool: {block.name}</Badge>
      <pre className="overflow-x-auto whitespace-pre-wrap break-words">
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
  onOutputChange?: (output: JsonInferenceOutput) => void;
}

function JsonOutput({ output, isEditing, onOutputChange }: JsonOutputProps) {
  const handleRawChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onOutputChange) {
      onOutputChange({
        ...output,
        raw: e.target.value,
      });
    }
  };

  return (
    <div className="space-y-4">
      {output.parsed && (
        <div className="rounded-md bg-muted p-4">
          <Badge className="mb-2">Parsed Output</Badge>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">
              {JSON.stringify(output.parsed, null, 2)}
            </code>
          </pre>
        </div>
      )}
      <div className="rounded-md bg-muted p-4">
        <Badge className="mb-2">Raw Output</Badge>
        {isEditing ? (
          <textarea
            className="w-full rounded border border-slate-300 p-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-800"
            value={output.raw}
            onChange={handleRawChange}
            rows={5}
          />
        ) : (
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">{output.raw}</code>
          </pre>
        )}
      </div>
    </div>
  );
}

// ContentBlocksOutput component for handling ContentBlockOutput[]
interface ContentBlocksOutputProps {
  blocks: ContentBlockOutput[];
  isEditing?: boolean;
  onBlocksChange?: (blocks: ContentBlockOutput[]) => void;
}

function ContentBlocksOutput({
  blocks,
  isEditing,
  onBlocksChange,
}: ContentBlocksOutputProps) {
  const handleBlockChange = (
    index: number,
    updatedBlock: ContentBlockOutput,
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
  const handleJsonOutputChange = (updatedOutput: JsonInferenceOutput) => {
    if (onOutputChange) {
      onOutputChange(updatedOutput);
    }
  };

  const handleBlocksChange = (updatedBlocks: ContentBlockOutput[]) => {
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
