import { Badge } from "../ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

/* This component displays a list of tag keys and values as badges.
 * They will be truncated if long, so we offer a tooltop on hover with the full value.
 */

interface TagsBadgesProps {
  tags: Record<string, string>;
}

export function TagsBadges({ tags }: TagsBadgesProps) {
  const tagEntries = Object.entries(tags);

  if (tagEntries.length === 0) {
    return <span className="text-muted-foreground text-sm">—</span>;
  }

  return (
    <TooltipProvider>
      <div className="flex max-w-[200px] flex-wrap gap-1">
        {tagEntries.map(([key, value]) => {
          const isSystemTag = key.startsWith("tensorzero::");
          const displayText = `${key}=${value}`;
          const truncatedText =
            displayText.length > 20
              ? `${displayText.substring(0, 17)}...`
              : displayText;

          return (
            <Tooltip key={key}>
              <TooltipTrigger asChild>
                <Badge
                  variant={isSystemTag ? "secondary" : "outline"}
                  className="cursor-help font-mono text-xs"
                >
                  {truncatedText}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <div className="max-w-xs break-words font-mono">
                  <strong>{key}</strong>={value}
                </div>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
