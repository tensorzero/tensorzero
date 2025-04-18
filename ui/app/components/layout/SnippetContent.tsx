import { Badge } from "~/components/ui/badge";
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

function EmptyMessage({ message = "No content defined" }: EmptyMessageProps) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-16 text-sm">
      {message}
    </div>
  );
}

// Label component
interface LabelProps {
  text?: string;
}

function Label({ text }: LabelProps) {
  if (!text) return null;

  return (
    <Badge className="bg-bg-muted text-fg-primary mx-4 mt-4 mb-0">{text}</Badge>
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

      <div className="bg-bg-primary w-full overflow-hidden rounded-lg">
        <div className="w-full">
          <div className="flex w-full">
            {showLineNumbers && (
              <div className="bg-bg-primary text-fg-muted pointer-events-none sticky left-0 min-w-[3rem] shrink-0 py-5 pr-3 pl-4 text-right font-mono select-none">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i} className="text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="w-0 grow overflow-auto">
              <pre className="w-full px-4 py-5">
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
  emptyMessage?: string;
}

export function TextMessage({
  label,
  content,
  emptyMessage,
}: TextMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  return (
    <div className="relative w-full">
      <Label text={label} />

      <div className="bg-bg-primary w-full overflow-hidden rounded-lg">
        <div className="p-5">
          <div className="text-fg-primary text-sm break-words whitespace-pre-wrap">
            {content || ""}
          </div>
        </div>
      </div>
    </div>
  );
}

// Input message component
interface InputMessageProps {
  role?: string;
  children?: React.ReactNode;
  emptyMessage?: string;
}

export function InputMessage({
  role,
  children,
  emptyMessage,
}: InputMessageProps) {
  if (!children) {
    return <EmptyMessage message={emptyMessage} />;
  }

  return (
    <div className="relative w-full">
      <div className="bg-bg-primary flex w-full flex-col gap-1 overflow-hidden rounded-lg px-5 py-2">
        <div className="text-sm font-medium text-purple-700 capitalize">
          {role}
        </div>
        <div className="my-1 flex">
          <div className="border-border mr-4 self-stretch border-l"></div>
          <div className="flex flex-1 flex-col gap-4">{children}</div>
        </div>
      </div>
    </div>
  );
}

// Input Text Message component
interface InputTextMessageProps {
  content: string;
}

export function InputTextMessage({ content }: InputTextMessageProps) {
  return (
    <div className="flex max-w-200 min-w-80 flex-col gap-1">
      <div className="flex flex-row items-center gap-1">
        <AlignLeft className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">Text</span>
      </div>
      <pre className="whitespace-pre-wrap">
        <span className="font-sans text-sm">{content}</span>
      </pre>
    </div>
  );
}

// Structured Text Message component
interface StructuredTextMessageProps {
  content: object;
}

export function StructuredTextMessage({ content }: StructuredTextMessageProps) {
  return (
    <div className="flex max-w-200 min-w-80 flex-col gap-1.5">
      <div className="flex flex-row items-center gap-1">
        <AlignLeft className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">
          Text (Structured)
        </span>
      </div>
      <pre className="max-w-full font-mono text-sm break-words whitespace-pre-wrap">
        {JSON.stringify(content, null, 2)}
      </pre>
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
    <div className="flex max-w-200 min-w-80 flex-col gap-1 overflow-x-auto">
      <div className="flex flex-row items-center gap-1">
        <Terminal className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">Tool Call</span>
      </div>
      <div className="border-border bg-bg-tertiary flex flex-col gap-1 rounded-md border px-3 py-2">
        <div className="flex flex-row items-center gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">Name:</span>
          <span className="text-sm">{toolName}</span>
        </div>
        <div className="flex flex-row items-center gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">ID:</span>
          <span className="font-mono text-sm">{toolCallId}</span>
        </div>
        <div className="flex flex-row items-start gap-1">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">Args:</span>
          <pre className="max-w-full font-mono text-sm break-words whitespace-pre-wrap">
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
    <div className="flex max-w-200 min-w-80 flex-col gap-1 overflow-x-auto">
      <div className="flex flex-row items-center gap-1">
        <ArrowRight className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">
          Tool Result
        </span>
      </div>
      <div className="border-border bg-bg-tertiary flex flex-col gap-1 rounded-md border px-3 py-2">
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">Name:</span>
          <span className="text-sm">{toolName}</span>
        </div>
        <div className="flex flex-row items-start gap-1 whitespace-nowrap">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">ID:</span>
          <span className="font-mono text-sm">{toolResultId}</span>
        </div>
        <div className="flex flex-row items-start gap-1">
          <span className="text-fg-secondary w-16 min-w-16 text-sm">
            Result:
          </span>
          <pre className="max-w-full font-mono text-sm break-words whitespace-pre-wrap">
            {toolResult}
          </pre>
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
      <div className="flex flex-row items-center gap-1">
        <ImageIcon className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">Image</span>
      </div>
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

// Image Error Message component
export function ImageErrorMessage() {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-row items-center gap-1">
        <ImageIcon className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">
          Image (Error)
        </span>
      </div>
      <div className="border-border bg-bg-tertiary relative aspect-video w-60 min-w-60 rounded-md border">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-2">
          <ImageOff className="text-fg-muted h-4 w-4" />
          <span className="text-fg-tertiary text-center text-xs font-medium">
            Failed to retrieve image
          </span>
        </div>
      </div>
    </div>
  );
}

// Raw Text Message component
interface RawTextMessageProps {
  content: string;
}

export function RawTextMessage({ content }: RawTextMessageProps) {
  return (
    <div className="flex max-w-200 min-w-80 flex-col gap-1.5">
      <div className="flex flex-row items-center gap-1">
        <AlignLeft className="text-fg-muted h-3 w-3" />
        <span className="text-fg-tertiary text-xs font-medium">Text (Raw)</span>
      </div>
      <pre className="whitespace-pre-wrap">
        <span className="font-mono text-sm">{content}</span>
      </pre>
    </div>
  );
}
