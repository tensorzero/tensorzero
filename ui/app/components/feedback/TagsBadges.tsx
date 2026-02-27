import { Badge } from "../ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

/* This component displays a list of tag keys and values as badges.
 * They will be truncated if long, so we offer a tooltop on hover with the full value.
 */

const MAX_VISIBLE_TAGS = 3;

interface TagsBadgesProps {
  tags: Record<string, string | undefined>;
}

function TagBadge({ tagKey, value }: { tagKey: string; value: string }) {
  const isSystemTag = tagKey.startsWith("tensorzero::");
  const displayText = `${tagKey}=${value}`;
  const truncatedText =
    displayText.length > 20
      ? `${displayText.substring(0, 17)}...`
      : displayText;

  return (
    <Tooltip>
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
          <strong>{tagKey}</strong>={value}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

export function TagsBadges({ tags }: TagsBadgesProps) {
  const tagEntries = Object.entries(tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );

  if (tagEntries.length === 0) {
    return <span className="text-muted-foreground text-sm">—</span>;
  }

  const visibleTags = tagEntries.slice(0, MAX_VISIBLE_TAGS);
  const hiddenCount = tagEntries.length - MAX_VISIBLE_TAGS;

  return (
    <div className="flex max-w-[200px] flex-wrap gap-1">
      {visibleTags.map(([key, value]) => (
        <TagBadge key={key} tagKey={key} value={value} />
      ))}
      {hiddenCount > 0 && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              variant="outline"
              className="text-muted-foreground cursor-help text-xs"
            >
              +{hiddenCount} more
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <div className="flex max-w-sm flex-col gap-1">
              {tagEntries.slice(MAX_VISIBLE_TAGS).map(([key, value]) => (
                <div key={key} className="font-mono text-xs break-words">
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
