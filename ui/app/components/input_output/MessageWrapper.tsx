import { type ReactNode } from "react";
import { clsx } from "clsx";

interface MessageWrapperProps {
  children?: ReactNode;
  role: "system" | "user" | "assistant";
  actionBar?: ReactNode;
}

/**
 * A wrapper component for displaying chat messages.
 * Used to wrap `SystemElement` and `MessageElement` components with consistent styling and layout.
 */
export default function MessageWrapper({
  children,
  role,
  actionBar,
}: MessageWrapperProps) {
  const labelClassName = clsx("text-sm font-medium capitalize", {
    "text-purple-500": role === "system",
    "text-emerald-500": role === "assistant",
    "text-blue-500": role === "user",
  });

  const messageContainerClassName = clsx(
    "flex w-full flex-col gap-4 border-l-2 pl-2",
    {
      "border-purple-200": role === "system",
      "border-emerald-200": role === "assistant",
      "border-blue-200": role === "user",
    },
  );

  return (
    <div className="flex w-full flex-col gap-1" data-testid={`message-${role}`}>
      <div className="relative flex items-center gap-1">
        <div className={labelClassName}>{role}</div>
        {actionBar && <div>{actionBar}</div>}
      </div>
      <div className={messageContainerClassName}>{children}</div>
    </div>
  );
}
