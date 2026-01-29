import type { LucideIcon } from "lucide-react";
import { cn } from "~/utils/common";

export enum AutopilotStatusBannerVariant {
  Warning = "warning",
  Error = "error",
}

const variantStyles = {
  [AutopilotStatusBannerVariant.Warning]:
    "border-amber-200 bg-amber-50 text-amber-800",
  [AutopilotStatusBannerVariant.Error]: "border-red-200 bg-red-50 text-red-800",
};

interface AutopilotStatusBannerProps {
  variant: AutopilotStatusBannerVariant;
  icon?: LucideIcon;
  children: React.ReactNode;
  className?: string;
}

export function AutopilotStatusBanner({
  variant,
  icon: Icon,
  children,
  className,
}: AutopilotStatusBannerProps) {
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
