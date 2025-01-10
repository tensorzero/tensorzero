import { useState } from "react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";

interface SFTResultProps {
  finalResult: string | null;
}

export function SFTResult({ finalResult }: SFTResultProps) {
  const [copied, setCopied] = useState(false);

  if (!finalResult) return null;

  return (
    <div className="mt-4 rounded-lg bg-gray-100 p-4">
      <div className="mb-2 flex items-center justify-between font-medium">
        <span>Configuration</span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            navigator.clipboard.writeText(finalResult);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          }}
        >
          {copied ? "Copied!" : "Copy to Clipboard"}
        </Button>
      </div>
      <Textarea
        value={finalResult}
        className="h-48 w-full resize-none border-none bg-transparent focus:ring-0"
        readOnly
      />
    </div>
  );
}
