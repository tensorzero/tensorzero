import { cn } from "~/utils/common";
import { type ReactNode } from "react";

interface ActionBarProps {
  children: ReactNode;
  className?: string;
}

export function ActionBar({ children, className }: ActionBarProps) {
  return <div className={cn("flex gap-3", className)}>{children}</div>;
}
