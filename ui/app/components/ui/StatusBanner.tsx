import type { LucideIcon } from "lucide-react";
import { cn } from "~/utils/common";

export enum StatusBannerVariant {
  Warning = "warning",
  Error = "error",
}

const variantStyles = {
  [StatusBannerVariant.Warning]: "border-amber-200 bg-amber-50 text-amber-800",
  [StatusBannerVariant.Error]: "border-red-200 bg-red-50 text-red-800",
};

interface StatusBannerProps {
  variant: StatusBannerVariant;
  icon?: LucideIcon;
  children: React.ReactNode;
  className?: string;
}

export function StatusBanner({
  variant,
  icon: Icon,
  children,
  className,
}: StatusBannerProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md border px-4 py-2 text-sm",
        variantStyles[variant],
        className,
      )}
    >
      {Icon && <Icon className="h-4 w-4 flex-shrink-0" />}
      <span>{children}</span>
    </div>
  );
}
