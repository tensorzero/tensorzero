import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface LegacyStructuredPromptBadgeProps {
  name: string;
  type: "template" | "schema";
}

export function LegacyStructuredPromptBadge({
  name,
  type,
}: LegacyStructuredPromptBadgeProps) {
  const legacyName = `${name}_${type}`;
  const newName = `${type}.${name}.path`;

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge className="bg-yellow-600 px-1 py-0 text-[10px] text-white">
            Legacy
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs p-2">
          <div className="text-xs">
            Please migrate from <code>{legacyName}</code> to{" "}
            <code>{newName}</code>.{" "}
            <a
              href="https://www.tensorzero.com/docs/gateway/create-a-prompt-template#migrate-from-legacy-prompt-templates"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-gray-300"
            >
              Read more
            </a>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
