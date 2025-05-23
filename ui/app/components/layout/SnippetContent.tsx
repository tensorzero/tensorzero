import { Badge } from "~/components/ui/badge";

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
    <Badge className="bg-bg-muted text-fg-primary mx-4 mb-0 mt-4">{text}</Badge>
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
              <div className="bg-bg-primary text-fg-muted pointer-events-none sticky left-0 min-w-[3rem] shrink-0 select-none py-5 pl-4 pr-3 text-right font-mono">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i} className="text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="w-0 grow overflow-auto">
              <pre className="w-full px-4 py-5">
                <code className="text-fg-primary block whitespace-pre font-mono text-sm leading-6">
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
          <div className="text-fg-primary whitespace-pre-wrap break-words text-sm">
            {content || ""}
          </div>
        </div>
      </div>
    </div>
  );
}
