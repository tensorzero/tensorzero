import { useEffect } from "react";
import { ClipboardIcon } from "lucide-react";
import type { Input, StoredInference } from "~/types/tensorzero";
import { useCopy } from "~/hooks/use-copy";
import { useToast } from "~/hooks/use-toast";
import { Button, ButtonIcon } from "~/components/ui/button";

interface CopyMessagesButtonProps {
  input: Input | undefined;
  output: StoredInference["output"];
}

export function CopyMessagesButton({ input, output }: CopyMessagesButtonProps) {
  const { copy, didCopy } = useCopy();
  const { toast } = useToast();

  useEffect(() => {
    if (didCopy) {
      toast.success({ title: "Copied messages to clipboard" });
    }
  }, [didCopy, toast]);

  const handleCopy = async () => {
    await copy(JSON.stringify({ input, output }, null, 2));
  };

  return (
    <Button variant="outline" size="sm" className="w-fit" onClick={handleCopy}>
      <ButtonIcon as={ClipboardIcon} variant="tertiary" />
      Copy Messages
    </Button>
  );
}
