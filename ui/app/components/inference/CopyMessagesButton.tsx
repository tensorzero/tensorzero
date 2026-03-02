import { useEffect } from "react";
import { ClipboardIcon } from "lucide-react";
import type { StoredInference, Input } from "~/types/tensorzero";
import { useCopy } from "~/hooks/use-copy";
import { useToast } from "~/hooks/use-toast";
import { Button, ButtonIcon } from "~/components/ui/button";

interface CopyMessagesButtonProps {
  input: Input | undefined | PromiseLike<Input | undefined>;
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
    const resolvedInput = await input;
    const data = { input: resolvedInput, output };
    await copy(JSON.stringify(data, null, 2));
  };

  return (
    <Button variant="outline" size="sm" className="w-fit" onClick={handleCopy}>
      <ButtonIcon as={ClipboardIcon} variant="tertiary" />
      Copy Messages
    </Button>
  );
}
