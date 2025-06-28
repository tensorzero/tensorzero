import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { useCopy } from "~/hooks/use-copy";

interface SFTResultProps {
  finalResult: string | null;
  forceShowCopyButton?: boolean;
}

export function SFTResult({
  finalResult,
  forceShowCopyButton = false,
}: SFTResultProps) {
  const { copy, didCopy, isCopyAvailable } = useCopy();

  return (
    finalResult && (
      <div className="mt-4 rounded-lg bg-gray-100 p-4">
        <div className="mb-2 flex min-h-8 items-center justify-between font-medium">
          <span>Configuration</span>
          {(isCopyAvailable || forceShowCopyButton) && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => copy(finalResult)}
            >
              {didCopy ? "Copied!" : "Copy to Clipboard"}
            </Button>
          )}
        </div>

        <Textarea
          value={finalResult}
          className="h-48 w-full resize-none border-none bg-transparent focus:ring-0"
          readOnly
        />
      </div>
    )
  );
}
