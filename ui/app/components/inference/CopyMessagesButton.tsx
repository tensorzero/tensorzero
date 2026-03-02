import { useEffect } from "react";
import { BracesIcon, ClipboardIcon, FileTextIcon } from "lucide-react";
import type { Input, StoredInference } from "~/types/tensorzero";
import { useCopy } from "~/hooks/use-copy";
import { useToast } from "~/hooks/use-toast";
import { Button, ButtonIcon } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { serializeConversationMarkdown } from "~/utils/serialize-markdown";

interface CopyMessagesButtonProps {
  input: Input | undefined;
  output: StoredInference["output"];
}

export function CopyMessagesButton({ input, output }: CopyMessagesButtonProps) {
  const { copy, didCopy } = useCopy();
  const { toast } = useToast();

  useEffect(() => {
    if (didCopy) {
      toast.success({ title: "Copied to clipboard" });
    }
  }, [didCopy, toast]);

  const handleCopyJson = async () => {
    await copy(JSON.stringify({ input, output }, null, 2));
  };

  const handleCopyMarkdown = async () => {
    await copy(serializeConversationMarkdown(input, output));
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="w-fit">
          <ButtonIcon as={ClipboardIcon} variant="tertiary" />
          Copy Messages
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem onClick={handleCopyJson}>
          <BracesIcon />
          JSON
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleCopyMarkdown}>
          <FileTextIcon />
          Markdown
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
