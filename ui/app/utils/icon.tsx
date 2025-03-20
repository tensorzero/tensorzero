import { TypeChat, TypeJson } from "~/components/icons/Icons";
import type { ReactNode } from "react";

export type IconConfig = {
  icon: ReactNode;
  iconBg: string;
};

/**
 * Get icon configuration for a function type
 * @param functionType The type of function ("json", "chat", etc.)
 * @returns IconConfig with icon component and background class
 */
export function getFunctionTypeIcon(functionType: string): IconConfig {
  switch (functionType?.toLowerCase()) {
    case "chat":
    case "conversation":
      return {
        icon: <TypeChat className="text-fg-type-chat" />,
        iconBg: "bg-bg-type-chat",
      };
    default:
      return {
        icon: <TypeJson className="text-fg-type-json" />,
        iconBg: "bg-bg-type-json",
      };
  }
}
