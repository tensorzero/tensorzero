import { type ReactNode } from "react";

interface ContentBlockLabelProps {
  icon?: ReactNode;
  children?: ReactNode;
  actionBar?: ReactNode;
}

export function ContentBlockLabel({
  icon,
  children,
  actionBar,
}: ContentBlockLabelProps) {
  return (
    children && (
      <div className="flex min-w-0 flex-row items-center gap-1">
        {icon}
        <div className="text-fg-tertiary min-w-0 text-xs font-medium">
          {children}
        </div>
        {actionBar}
      </div>
    )
  );
}
