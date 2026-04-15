import { Badge } from "../ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

/* This component displays a list of tag keys and values as badges.
 * They will be truncated if long, so we offer a tooltop on hover with the full value.
 * When there are more than MAX_VISIBLE_TAGS, extra tags collapse into a "+N more" badge.
 */

const MAX_VISIBLE_TAGS = 3;

interface TagsBadgesProps {
  tags: Record<string, string | undefined>;
}

export function TagsBadges({ tags }: TagsBadgesProps) {
  const tagEntries = Object.entries(tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );

  if (tagEntries.length === 0) {
    return <span className="text-muted-foreground text-sm">—</span>;
  }

  const visibleTags = tagEntries.slice(0, MAX_VISIBLE_TAGS);
  const hiddenTags = tagEntries.slice(MAX_VISIBLE_TAGS);

  return (
    <div className="flex max-w-[200px] flex-wrap gap-1">
      {visibleTags.map(([key, value]) => {
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
              <div className="max-w-xs font-mono break-words">
                <strong>{key}</strong>={value}
              </div>
            </TooltipContent>
          </Tooltip>
        );
      })}
      {hiddenTags.length > 0 && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="cursor-help text-xs">
              +{hiddenTags.length} more
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <div className="max-w-xs space-y-1 font-mono break-words">
              {hiddenTags.map(([key, value]) => (
                <div key={key}>
                  <strong>{key}</strong>={value}
                </div>
              ))}
            </div>
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}
