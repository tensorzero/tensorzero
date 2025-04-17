import { Badge } from "~/components/ui/badge";
import { Link } from "react-router";

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
  role: string;
  children: React.ReactNode;
}

export function InputMessage({ role, children }: InputMessageProps) {
  return (
    <div className="relative w-full">
      <div className="bg-bg-primary flex w-full flex-col gap-1 overflow-hidden rounded-lg px-5 py-2">
        <div className="text-sm font-medium text-purple-700 capitalize">
          {role}
        </div>
        <div className="flex">
          <div className="border-border mr-4 self-stretch border-l"></div>
          <div className="flex-1">{children}</div>
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
    <pre className="whitespace-pre-wrap">
      <span className="font-sans text-sm">{content}</span>
    </pre>
  );
}

// Tool Call Message component
interface ToolCallMessageProps {
  label: string;
  content: string;
}

export function ToolCallMessage({ label, content }: ToolCallMessageProps) {
  return (
    <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
      <span className="font-medium">{label}</span>
      <pre className="mt-1 text-sm">{content}</pre>
    </div>
  );
}

// Tool Result Message component
interface ToolResultMessageProps {
  label: string;
  content: string;
}

export function ToolResultMessage({ label, content }: ToolResultMessageProps) {
  return (
    <div className="rounded bg-slate-100 p-2 dark:bg-slate-800">
      <span className="font-medium">{label}</span>
      <pre className="mt-1 text-sm">{content}</pre>
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
    <div className="w-60 rounded bg-slate-100 p-2 text-xs text-slate-300">
      <div className="mb-2">Image</div>
      <Link
        to={url}
        target="_blank"
        rel="noopener noreferrer"
        download={downloadName}
      >
        <img src={url} alt="Image" />
      </Link>
    </div>
  );
}

// Image Error Message component
export function ImageErrorMessage() {
  return (
    <div className="relative aspect-square w-[150px] rounded-md bg-slate-200">
      <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
        <span className="text-center text-sm text-balance text-red-500/40">
          Failed to retrieve image.
        </span>
      </div>
    </div>
  );
}
