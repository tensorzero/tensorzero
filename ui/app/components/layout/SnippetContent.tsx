import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Copy } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "~/utils/common";

// Code content component
interface CodeMessageProps {
  label?: string;
  content?: string;
  showLineNumbers?: boolean;
  forceShowCopyButton?: boolean;
  className?: string;
}

export function CodeMessage({
  label,
  content,
  showLineNumbers = false,
  forceShowCopyButton = false,
  className,
}: CodeMessageProps) {
  if (!content) return null;
  const [error, setError] = useState<string | null>(null);

  const isSecureContext =
    typeof window !== "undefined" && window.isSecureContext;
  const isClipboard = typeof navigator !== "undefined" && !!navigator.clipboard;
  const shouldShowCopyButton =
    (isSecureContext && isClipboard) || forceShowCopyButton;
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to copy. Please copy manually.");
      if (textAreaRef.current) {
        textAreaRef.current.select();
      }
    }
  };

  const lines = content.split("\n");

  return (
    <div className={cn("relative w-full max-w-full", className)}>
      {label && <Badge className="mb-2">{label}</Badge>}

      <div className="relative w-full max-w-full overflow-hidden rounded-lg bg-bg-primary">
        {shouldShowCopyButton && (
          <Button
            variant="outline"
            size="icon"
            onClick={handleCopy}
            className="absolute right-2 top-2 z-10 h-7 w-7 p-0 shadow-none"
            aria-label="Copy code"
          >
            <Copy className="h-4 w-4" />
          </Button>
        )}

        <textarea
          ref={textAreaRef}
          value={content}
          readOnly
          className="sr-only"
          aria-hidden="true"
        />

        {error && (
          <div className="absolute right-2 top-10 z-10 mt-2 rounded bg-red-100 p-2 text-xs text-red-800">
            {error}
          </div>
        )}

        <div className="relative w-full max-w-full overflow-hidden">
          <div className="flex max-w-full">
            {showLineNumbers && (
              <div className="flex-shrink-0 select-none pb-4 pl-4 pr-3 pt-4 text-right font-mono text-fg-muted">
                {lines.map((_, i) => (
                  <div key={i} className="h-[1.5rem] text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="min-w-0 flex-1 overflow-auto">
              <pre className="w-full max-w-full overflow-x-auto p-4">
                <code className="block font-mono text-sm leading-6 text-fg-primary">
                  {lines.map((line, i) => (
                    <div key={i} className="h-[1.5rem] truncate">
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
  forceShowCopyButton?: boolean;
  className?: string;
}

export function TextMessage({
  label,
  content,
  forceShowCopyButton = false,
  className,
}: TextMessageProps) {
  if (!content) return null;
  const [error, setError] = useState<string | null>(null);

  const isSecureContext =
    typeof window !== "undefined" && window.isSecureContext;
  const isClipboard = typeof navigator !== "undefined" && !!navigator.clipboard;
  const shouldShowCopyButton =
    (isSecureContext && isClipboard) || forceShowCopyButton;
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to copy. Please copy manually.");
      if (textAreaRef.current) {
        textAreaRef.current.select();
      }
    }
  };

  return (
    <div className={cn("relative w-full max-w-full", className)}>
      {label && <Badge className="mb-2">{label}</Badge>}

      <div className="relative w-full max-w-full overflow-hidden rounded-lg bg-bg-primary">
        {shouldShowCopyButton && (
          <Button
            variant="outline"
            size="icon"
            onClick={handleCopy}
            className="absolute right-2 top-2 z-10 h-7 w-7 p-0 shadow-none"
            aria-label="Copy text"
          >
            <Copy className="h-4 w-4" />
          </Button>
        )}

        <textarea
          ref={textAreaRef}
          value={content}
          readOnly
          className="sr-only"
          aria-hidden="true"
        />

        {error && (
          <div className="absolute right-2 top-10 z-10 mt-2 rounded bg-red-100 p-2 text-xs text-red-800">
            {error}
          </div>
        )}

        <div className="relative w-full max-w-full overflow-hidden">
          <div className="p-4">
            <div className="max-w-full overflow-x-auto whitespace-pre-wrap break-words text-fg-primary">
              {content}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
