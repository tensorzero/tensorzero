import { Card, CardContent } from "~/components/ui/card";
import type {
  InputMessageContent,
  ResolvedInput,
  ResolvedInputMessage,
  ResolvedInputMessageContent,
} from "~/utils/clickhouse/common";
import { SystemContent } from "./SystemContent";
import { useEffect, useState } from "react";
import { SkeletonImage } from "./SkeletonImage";
import ImageBlock from "./ImageBlock";

// Base interface with just the common required properties
interface BaseInputProps {
  input: ResolvedInput;
}

// For when isEditing is not provided (default behavior)
interface DefaultInputProps extends BaseInputProps {
  isEditing?: never;
  onSystemChange?: never;
  onMessagesChange?: never;
}

// For when isEditing is explicitly provided
interface EditableInputProps extends BaseInputProps {
  isEditing: boolean;
  onSystemChange: (system: string | object) => void;
  onMessagesChange: (messages: ResolvedInputMessage[]) => void;
}

type InputProps = DefaultInputProps | EditableInputProps;

export default function Input({
  input,
  isEditing,
  onSystemChange,
  onMessagesChange,
}: InputProps) {
  const handleMessageChange = (
    index: number,
    updatedMessage: ResolvedInputMessage,
  ) => {
    const updatedMessages = [...input.messages];
    updatedMessages[index] = updatedMessage;
    onMessagesChange?.(updatedMessages);
  };

  return (
    <Card>
      <CardContent className="space-y-6 pt-6">
        {(input.system || isEditing) && (
          <SystemContent
            systemContent={input.system}
            isEditing={isEditing ?? false}
            onChange={onSystemChange}
          />
        )}

        <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
          <div className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
            Messages
          </div>
          <div className="space-y-4">
            {input.messages.map((message, index) => (
              <Message
                key={index}
                message={message}
                isEditing={isEditing ?? false}
                onMessageChange={(updatedMessage) =>
                  handleMessageChange(index, updatedMessage)
                }
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface BaseMessageProps {
  message: ResolvedInputMessage;
}

interface DefaultMessageProps extends BaseMessageProps {
  isEditing?: never;
  onMessageChange?: never;
}

interface EditableMessageProps extends BaseMessageProps {
  isEditing: boolean;
  onMessageChange: (message: ResolvedInputMessage) => void;
}

type MessageProps = DefaultMessageProps | EditableMessageProps;

function Message({ message, isEditing, onMessageChange }: MessageProps) {
  const handleContentChange = (
    updatedContent: ResolvedInputMessage["content"],
  ) => {
    onMessageChange?.({ ...message, content: updatedContent });
  };

  return (
    <div className="space-y-1">
      <div className="text-md font-medium text-slate-600 capitalize dark:text-slate-400">
        {message.role}
      </div>
      <MessageContent
        content={message.content}
        isEditing={isEditing ?? false}
        onContentChange={handleContentChange}
      />
    </div>
  );
}

interface BaseMessageContentProps {
  content: ResolvedInputMessage["content"];
}

interface DefaultMessageContentProps extends BaseMessageContentProps {
  isEditing?: never;
  onContentChange?: never;
}

interface EditableMessageContentProps extends BaseMessageContentProps {
  isEditing: boolean;
  onContentChange: (content: ResolvedInputMessage["content"]) => void;
}

type MessageContentProps =
  | DefaultMessageContentProps
  | EditableMessageContentProps;

function MessageContent({
  content,
  isEditing,
  onContentChange,
}: MessageContentProps) {
  const handleBlockChange = (
    index: number,
    updatedBlock: ResolvedInputMessageContent,
  ) => {
    const updatedContent = [...content];
    updatedContent[index] = updatedBlock;
    onContentChange?.(updatedContent);
  };
  return (
    <div className="space-y-2">
      {content.map((block, index) => {
        switch (block.type) {
          case "text":
            return (
              <TextBlock
                key={index}
                block={block}
                isEditing={isEditing ?? false}
                onContentChange={(updatedBlock) =>
                  handleBlockChange(index, updatedBlock)
                }
              />
            );
          case "tool_call":
            return (
              <InputToolCallBlock
                key={index}
                block={block}
                isEditing={isEditing ?? false}
                onContentChange={(updatedBlock) =>
                  handleBlockChange(index, updatedBlock)
                }
              />
            );
          case "tool_result":
            return (
              <ToolResultBlock
                key={index}
                block={block}
                isEditing={isEditing ?? false}
                onContentChange={(updatedBlock) =>
                  handleBlockChange(index, updatedBlock)
                }
              />
            );
          case "file":
            if (block.file.mime_type.startsWith("image/")) {
              return <ImageBlock key={index} image={block} />;
            } else {
              return (
                <div key={index}>
                  <SkeletonImage
                    error={`Unsupported file type: ${block.file.mime_type}`}
                  />
                </div>
              );
            }
          case "file_error":
            return (
              <div key={index}>
                <SkeletonImage error="Failed to retrieve image." />
              </div>
            );
          default:
            return null;
        }
      })}
    </div>
  );
}

// TextBlock Component
interface TextBlockProps {
  block: Extract<InputMessageContent, { type: "text" }>;
  isEditing?: boolean;
  onContentChange?: (
    block: Extract<InputMessageContent, { type: "text" }>,
  ) => void;
}

function TextBlock({ block, isEditing, onContentChange }: TextBlockProps) {
  const [isObject, setIsObject] = useState(typeof block.value === "object");
  const [displayValue, setDisplayValue] = useState(
    isObject ? JSON.stringify(block.value, null, 2) : block.value,
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onContentChange) {
      const newValue = e.target.value;
      setDisplayValue(newValue);

      if (isObject) {
        try {
          const parsedValue = JSON.parse(newValue);
          setIsObject(typeof parsedValue === "object");
          setJsonError(null);
          onContentChange({
            type: "text",
            value: parsedValue,
          });
        } catch {
          setJsonError("Invalid JSON format");
        }
      } else {
        onContentChange({
          type: "text",
          value: newValue,
        });
      }
    }
  };

  if (isEditing) {
    return (
      <div className="w-full">
        <textarea
          className={`w-full rounded border bg-white p-2 font-mono text-sm dark:bg-slate-800 ${
            jsonError
              ? "border-red-500 dark:border-red-500"
              : "border-slate-300 dark:border-slate-700"
          }`}
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
    <pre className="whitespace-pre-wrap">
      <code className="text-sm">
        {typeof block.value === "object"
          ? JSON.stringify(block.value, null, 2)
          : block.value}
      </code>
    </pre>
  );
}

interface InputToolCallBlockProps {
  block: Extract<InputMessageContent, { type: "tool_call" }>;
  isEditing?: boolean;
  onContentChange?: (
    block: Extract<InputMessageContent, { type: "tool_call" }>,
  ) => void;
}

function InputToolCallBlock({
  block,
  isEditing,
  onContentChange,
}: InputToolCallBlockProps) {
  const [displayValue, setDisplayValue] = useState(block.arguments);
  const [jsonError, setJsonError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onContentChange) {
      const newValue = e.target.value;
      setDisplayValue(newValue);

      try {
        JSON.parse(newValue);
        setJsonError(null);
        onContentChange({
          type: "tool_call",
          name: block.name,
          arguments: newValue,
          id: block.id,
        });
      } catch {
        setJsonError("Invalid JSON format");
      }
    }
  };

  useEffect(() => {
    // Update display value when block.arguments changes externally
    setDisplayValue(block.arguments);
  }, [block.arguments]);

  if (isEditing) {
    return (
      <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
        <div className="font-medium">Tool: {block.name}</div>
        <textarea
          className={`mt-1 w-full rounded border bg-white p-2 font-mono text-sm ${
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
    <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
      <div className="font-medium">Tool: {block.name}</div>
      <pre className="mt-1 text-sm">{block.arguments}</pre>
    </div>
  );
}

// ToolResultBlock Component
interface ToolResultBlockProps {
  block: Extract<InputMessageContent, { type: "tool_result" }>;
  isEditing?: boolean;
  onContentChange?: (
    block: Extract<InputMessageContent, { type: "tool_result" }>,
  ) => void;
}

function ToolResultBlock({
  block,
  isEditing,
  onContentChange,
}: ToolResultBlockProps) {
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (onContentChange) {
      onContentChange({
        type: "tool_result",
        name: block.name,
        result: e.target.value,
        id: block.id,
      });
    }
  };

  if (isEditing) {
    return (
      <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
        <div className="font-medium">Result from: {block.name}</div>
        <textarea
          className="mt-1 w-full rounded border border-slate-300 bg-white p-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-800"
          value={block.result}
          onChange={handleChange}
          rows={3}
        />
      </div>
    );
  }

  return (
    <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
      <div className="font-medium">Result from: {block.name}</div>
      <pre className="mt-1 text-sm">{block.result}</pre>
    </div>
  );
}
