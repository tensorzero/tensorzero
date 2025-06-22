import { type ReactNode } from "react";

interface BasicInfoLayoutProps {
  children: ReactNode;
}

export function BasicInfoLayout({ children }: BasicInfoLayoutProps) {
  return <div className="flex flex-col gap-4 md:gap-2">{children}</div>;
}

interface BasicInfoItemProps {
  children: ReactNode;
}

export function BasicInfoItem({ children }: BasicInfoItemProps) {
  return <div className="flex flex-col gap-0.5 md:flex-row">{children}</div>;
}

interface BasicInfoItemTitleProps {
  children: ReactNode;
}

export function BasicInfoItemTitle({ children }: BasicInfoItemTitleProps) {
  return (
    <div className="text-fg-secondary w-full flex-shrink-0 text-left text-sm md:w-32 md:py-1">
      {children}
    </div>
  );
}

interface BasicInfoItemContentProps {
  children: ReactNode;
}

export function BasicInfoItemContent({ children }: BasicInfoItemContentProps) {
  return (
    <div className="text-fg-primary flex flex-wrap gap-x-4 gap-y-0.5 md:gap-1 md:py-1">
      {children}
    </div>
  );
}
