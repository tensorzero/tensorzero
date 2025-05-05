import { type ReactNode } from "react";

interface BasicInfoLayoutProps {
  children: ReactNode;
}

export function BasicInfoLayout({ children }: BasicInfoLayoutProps) {
  return <div className="flex flex-col gap-2">{children}</div>;
}

interface BasicInfoItemProps {
  children: ReactNode;
}

export function BasicInfoItem({ children }: BasicInfoItemProps) {
  return <div className="flex flex-row">{children}</div>;
}

interface BasicInfoItemTitleProps {
  children: ReactNode;
}

export function BasicInfoItemTitle({ children }: BasicInfoItemTitleProps) {
  return (
    <div className="text-fg-secondary w-32 py-1 text-left text-sm">
      {children}
    </div>
  );
}

interface BasicInfoItemContentProps {
  children: ReactNode;
}

export function BasicInfoItemContent({ children }: BasicInfoItemContentProps) {
  return (
    <div className="text-fg-primary ml-6 flex-1 text-left text-sm">
      {children}
    </div>
  );
}
