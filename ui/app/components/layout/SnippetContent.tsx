import { Link } from "react-router";
import {
  AlignLeft,
  Terminal,
  ArrowRight,
  Image as ImageIcon,
  ImageOff,
} from "lucide-react";

// Empty message component
interface EmptyMessageProps {
  message?: string;
}

export function EmptyMessage({
  message = "No content defined",
}: EmptyMessageProps) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
      {message}
    </div>
  );
}

// Label component
interface LabelProps {
  text?: string;
  icon?: React.ReactNode;
}

function Label({ text, icon }: LabelProps) {
  if (!text) return null;

  return (
    <div className="flex flex-row items-center gap-1">
      {icon && icon}
      <span className="text-fg-tertiary text-xs font-medium">{text}</span>
    </div>
  );
}

// Code content component
interface CodeMessageProps {
  label?: string;
  content?: string;
  showLineNumbers?: boolean;
  emptyMessage?: string;
}

export function CodeMessage({
  label,
  content,
  showLineNumbers = false,
  emptyMessage,
}: CodeMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  // We still need line count for line numbers, but won't split the content for rendering
  const lineCount = content ? content.split("\n").length : 0;

  return (
    <div className="relative w-full">
      <Label text={label} />

      <div className="w-full overflow-hidden">
        <div className="w-full">
          <div className="flex w-full">
            {showLineNumbers && (
              <div className="text-fg-muted pointer-events-none sticky left-0 min-w-[2rem] shrink-0 pr-3 text-right font-mono select-none">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i} className="text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="w-0 grow overflow-auto">
              <pre className="w-full">
                <code className="text-fg-primary block font-mono text-sm leading-6 whitespace-pre">
                  {content || ""}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Text content component
interface TextMessageProps {
  label?: string;
  content?: string;
  type?: "default" | "structured";
  emptyMessage?: string;
}

export function TextMessage({
  label,
  content,
  type = "default",
  emptyMessage,
}: TextMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text={label}
        icon={<AlignLeft className="text-fg-muted h-3 w-3" />}
      />
      {type === "structured" ? (
        <pre className="w-full font-mono text-sm break-words whitespace-pre-wrap">
          {content}
        </pre>
      ) : (
        <span className="text-fg-primary w-full text-sm">{content}</span>
      )}
    </div>
  );
}

// Tool Call Message component
interface ToolCallMessageProps {
  toolName: string;
  toolArguments: string;
  toolCallId: string;
}

export function ToolCallMessage({
  toolName,
  toolArguments,
  toolCallId,
}: ToolCallMessageProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text="Tool Call"
        icon={<Terminal className="text-fg-muted h-3 w-3" />}
      />
      <div className="border-border bg-bg-tertiary flex flex-col gap-1 rounded-md border px-3 py-2 text-sm">
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16">Name:</span>
          <span className="overflow-hidden text-ellipsis">{toolName}</span>
        </div>
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16">ID:</span>
          <span className="overflow-hidden font-mono text-ellipsis">
            {toolCallId}
          </span>
        </div>
        <div className="flex flex-row items-start gap-1">
          <span className="text-fg-secondary w-16 min-w-16">Args:</span>
          <pre className="max-w-full font-mono break-words whitespace-pre-wrap">
            {toolArguments}
          </pre>
        </div>
      </div>
    </div>
  );
}

// Tool Result Message component
interface ToolResultMessageProps {
  toolName: string;
  toolResult: string;
  toolResultId: string;
}

export function ToolResultMessage({
  toolName,
  toolResult,
  toolResultId,
}: ToolResultMessageProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text="Tool Result"
        icon={<ArrowRight className="text-fg-muted h-3 w-3" />}
      />
      <div className="border-border bg-bg-tertiary flex flex-col gap-1 rounded-md border px-3 py-2 text-sm">
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16">Name:</span>
          <span className="overflow-hidden text-ellipsis">{toolName}</span>
        </div>
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16">ID:</span>
          <span className="overflow-hidden font-mono text-ellipsis">
            {toolResultId}
          </span>
        </div>
        <div className="flex flex-row items-start gap-1">
          <span className="text-fg-secondary w-16 min-w-16">Result:</span>
          <div className="w-full overflow-x-auto">
            <pre className="max-w-full font-mono break-words whitespace-pre-wrap">
              {toolResult}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

// Image Message component
interface ImageMessageProps {
  url: string;
  downloadName?: string;
}

export function ImageMessage({ url, downloadName }: ImageMessageProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label
        text="Image"
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
      />
      <div>
        <Link
          to={url}
          target="_blank"
          rel="noopener noreferrer"
          download={downloadName}
          className="border-border bg-bg-tertiary text-fg-tertiary flex min-h-20 w-60 items-center justify-center rounded border p-2 text-xs"
        >
          <img src={url} alt="Image" />
        </Link>
      </div>
    </div>
  );
}

interface ImageErrorMessageProps {
  error: string;
}

// Image Error Message component
export function ImageErrorMessage({ error }: ImageErrorMessageProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label
        text="Image (Error)"
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
      />
      <div className="border-border bg-bg-tertiary relative aspect-video w-60 min-w-60 rounded-md border">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-2">
          <ImageOff className="text-fg-muted h-4 w-4" />
          <span className="text-fg-tertiary text-center text-xs font-medium">
            {error}
          </span>
        </div>
      </div>
    </div>
  );
}
