import { TypeChat, TypeJson, Check, Cross } from "~/components/icons/Icons";
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

/**
 * Get icon configuration for feedback status
 * @param status The status ("success", "failure", or "neutral")
 * @returns IconConfig with icon component and background class
 */
export function getFeedbackIcon(status: "success" | "failure" | "neutral"): IconConfig {
  switch (status) {
    case "success":
      return {
        icon: <Check className="text-green-800" size={16} />,
        iconBg: "bg-green-200",
      };
    case "failure":
      return {
        icon: <Cross className="text-red-800" size={16} />,
        iconBg: "bg-red-200",
      };
    case "neutral":
    default:
      return {
        icon: null,
        iconBg: "bg-gray-100",
      };
  }
}
