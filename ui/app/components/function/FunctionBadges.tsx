import { Badge } from "~/components/ui/badge";
import type { FunctionConfig } from "~/types/tensorzero";

const getBadgeStyle = (type: string) => {
  switch (type) {
    case "chat":
      return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300";
    case "json":
      return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
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
