import { useState, useRef } from "react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";

interface SFTResultProps {
  finalResult: string | null;
  forceShowCopyButton?: boolean;
}

export function SFTResult({
  finalResult,
  forceShowCopyButton = false,
}: SFTResultProps) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const isSecureContext =
    typeof window !== "undefined" && window.isSecureContext;
  const isClipboard = typeof navigator !== "undefined" && !!navigator.clipboard;
  const shouldShowCopyButton =
    (isSecureContext && isClipboard) || forceShowCopyButton;

  if (!finalResult) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(finalResult);
      setCopied(true);
      setError(null);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error(err);
      setError("Failed to copy. Please copy manually.");
      if (textAreaRef.current) {
        textAreaRef.current.select();
      }
    }
  };

  return (
    <div className="mt-4 rounded-lg bg-gray-100 p-4">
      <div className="mb-2 flex min-h-8 items-center justify-between font-medium">
        <span>Configuration</span>
        {shouldShowCopyButton ? (
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? "Copied!" : "Copy to Clipboard"}
          </Button>
        ) : null}
      </div>
      {error && <p className="mb-2 text-sm text-red-500">{error}</p>}
      <Textarea
        ref={textAreaRef}
        value={finalResult}
        className="h-48 w-full resize-none border-none bg-transparent focus:ring-0"
        readOnly
      />
    </div>
  );
}
