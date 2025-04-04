import { type LucideIcon } from "lucide-react";

interface MetadataItemProps {
  icon: LucideIcon;
  label: string;
  value: string;
  isRaw?: boolean;
}

export function MetadataItem({
  icon: Icon,
  label,
  value,
  isRaw = false,
}: MetadataItemProps) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4" />
      <span className="font-medium">{label}:</span>{" "}
      {isRaw ? (
        value
      ) : (
        <code className="bg-muted rounded px-1 py-0.5 text-sm">{value}</code>
      )}
    </div>
  );
}
