import { cn } from "~/utils/common";
import { type ReactNode } from "react";

interface BasicInfoLayoutProps {
  children: ReactNode;
  className?: string;
}

export function BasicInfoLayout({ children, className }: BasicInfoLayoutProps) {
  return <div className={cn("flex flex-col gap-2", className)}>{children}</div>;
}

interface BasicInfoItemProps {
  children: ReactNode;
  className?: string;
}

export function BasicInfoItem({ children, className }: BasicInfoItemProps) {
  return <div className={cn("flex flex-row", className)}>{children}</div>;
}

interface BasicInfoItemTitleProps {
  children: ReactNode;
  className?: string;
}

export function BasicInfoItemTitle({
  children,
  className,
}: BasicInfoItemTitleProps) {
  return (
    <div
      className={cn("text-fg-secondary w-32 py-1 text-left text-sm", className)}
    >
      {children}
    </div>
  );
}

interface BasicInfoItemContentProps {
  children: ReactNode;
  className?: string;
}

export function BasicInfoItemContent({
  children,
  className,
}: BasicInfoItemContentProps) {
  return (
    <div
      className={cn("text-fg-primary ml-6 flex-1 text-left text-sm", className)}
    >
      {children}
    </div>
  );
}
