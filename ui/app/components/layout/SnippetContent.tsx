import { Badge } from "~/components/ui/badge";
import { cn } from "~/utils/common";

// Empty message component
interface EmptyMessageProps {
  message?: string;
  className?: string;
}

export function EmptyMessage({
  message = "No content defined",
  className,
}: EmptyMessageProps) {
  return (
    <div
      className={cn(
        "text-fg-muted flex items-center justify-center py-16 text-sm",
        className,
      )}
    >
      {message}
    </div>
  );
}

// Label component
interface LabelProps {
  text?: string;
  className?: string;
}

export function Label({ text, className }: LabelProps) {
  if (!text) return null;

  return (
    <Badge
      className={cn("bg-bg-muted text-fg-primary mx-4 mt-4 mb-0", className)}
    >
      {text}
    </Badge>
  );
}

// Code content component
interface CodeMessageProps {
  label?: string;
  content?: string;
  showLineNumbers?: boolean;
  emptyMessage?: string;
  className?: string;
}

export function CodeMessage({
  label,
  content,
  showLineNumbers = false,
  emptyMessage,
  className,
}: CodeMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  // We still need line count for line numbers, but won't split the content for rendering
  const lineCount = content ? content.split("\n").length : 0;

  return (
    <div className={cn("relative w-full", className)}>
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
                  {content || " "}
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
  className?: string;
}

export function TextMessage({
  label,
  content,
  emptyMessage,
  className,
}: TextMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  return (
    <div className={cn("relative w-full", className)}>
      <Label text={label} />

      <div className="bg-bg-primary w-full overflow-hidden rounded-lg">
        <div className="p-5">
          <div className="text-fg-primary text-sm break-words whitespace-pre-wrap">
            {content}
          </div>
        </div>
      </div>
    </div>
  );
}
