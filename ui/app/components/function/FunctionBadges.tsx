import { Badge } from "~/components/ui/badge";
import type { FunctionConfig } from "~/types/tensorzero";

const getBadgeStyle = (type: string) => {
  switch (type) {
    case "chat":
      return "bg-blue-100 text-blue-800";
    case "json":
      return "bg-purple-100 text-purple-800";
    default:
      return "bg-bg-tertiary text-fg-primary";
  }
};

type FunctionBadgesProps = {
  fn: FunctionConfig;
};

export function FunctionBadges({ fn }: FunctionBadgesProps) {
  return (
    <Badge className={getBadgeStyle(fn.type)}>
      {fn.type === "chat" ? "Chat" : fn.type === "json" ? "JSON" : "Unknown"}
    </Badge>
  );
}
