import { type ReactNode } from "react";

interface ActionBarProps {
  children: ReactNode;
  className?: string;
}

export function ActionBar({ children }: ActionBarProps) {
  return <div className="flex gap-2">{children}</div>;
}
