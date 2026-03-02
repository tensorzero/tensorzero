import { Badge } from "../ui/badge";

interface TagsBadgesProps {
  tags: Record<string, string | undefined>;
}

function TagBadge({ tagKey, value }: { tagKey: string; value: string }) {
  const isSystemTag = tagKey.startsWith("tensorzero::");

  return (
    <Badge
      variant={isSystemTag ? "secondary" : "outline"}
      className="font-mono text-xs"
    >
      {tagKey}={value}
    </Badge>
  );
}

export function TagsBadges({ tags }: TagsBadgesProps) {
  const tagEntries = Object.entries(tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );

  if (tagEntries.length === 0) {
    return <span className="text-fg-muted text-sm">—</span>;
  }

  return (
    <div className="flex flex-wrap gap-1">
      {tagEntries.map(([key, value]) => (
        <TagBadge key={key} tagKey={key} value={value} />
      ))}
    </div>
  );
}
