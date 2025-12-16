import { TypeChat, TypeJson, QuestionMark } from "~/components/icons/Icons";
import type { ReactNode } from "react";

export type IconConfig = {
  icon: ReactNode;
  iconBg: string;
  label?: string;
};

/**
 * Get icon configuration for a function type
 * @param functionType The type of function ("json", "chat", etc.)
 * @returns IconConfig with icon component and background class
 */
export function getFunctionTypeIcon(functionType: string): IconConfig {
  switch (functionType?.toLowerCase()) {
    case "chat":
      return {
        icon: <TypeChat className="text-fg-type-chat" />,
        iconBg: "bg-bg-type-chat",
        label: "Function type: Chat",
      };
    case "json":
      return {
        icon: <TypeJson className="text-fg-type-json" />,
        iconBg: "bg-bg-type-json",
        label: "Function type: JSON",
      };
    default:
      return {
        icon: <QuestionMark className="text-neutral-600" />,
        iconBg: "bg-neutral-100",
        label: "Function type: Unknown",
      };
  }
}
