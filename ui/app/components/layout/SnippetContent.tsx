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
        "flex items-center justify-center py-16 text-sm text-fg-secondary",
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
      className={cn("mx-4 mb-0 mt-4 bg-bg-muted text-fg-primary", className)}
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

  const lines = content.split("\n");

  return (
    <div className={cn("relative w-full", className)}>
      <Label text={label} />

      <div className="w-full overflow-hidden rounded-lg bg-bg-primary">
        <div className="w-full">
          <div className="flex w-full">
            {showLineNumbers && (
              <div className="pointer-events-none sticky left-0 flex-shrink-0 select-none bg-bg-primary pb-4 pl-4 pr-3 pt-4 text-right font-mono text-fg-muted">
                {lines.map((_, i) => (
                  <div key={i} className="h-[1.5rem] text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="w-0 flex-grow overflow-auto">
              <pre className="min-w-full p-4">
                <code className="block whitespace-pre font-mono text-sm leading-6 text-fg-primary">
                  {lines.map((line, i) => (
                    <div key={i} className="h-[1.5rem]">
                      {line || " "}
                    </div>
                  ))}
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

      <div className="w-full overflow-hidden rounded-lg bg-bg-primary">
        <div className="p-4">
          <div className="whitespace-pre-wrap break-words text-sm text-fg-primary">
            {content}
          </div>
        </div>
      </div>
    </div>
  );
}
