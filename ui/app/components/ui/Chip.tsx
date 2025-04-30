import React from "react";
import { Link } from "react-router";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "~/components/ui/tooltip";

interface ChipProps {
  icon?: React.ReactNode;
  label: string;
  secondaryLabel?: string;
  link?: string;
  className?: string;
  font?: "sans" | "mono";
  tooltip?: React.ReactNode;
  iconBg?: string;
}

export const Chip: React.FC<ChipProps> = ({
  label,
  icon,
  secondaryLabel,
  link,
  className = "",
  font = "sans",
  tooltip,
  iconBg = "bg-none",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-fg-primary py-1 px-2 gap-1.5 rounded-md whitespace-nowrap overflow-hidden";
  const hoverClasses = link ? "hover:bg-bg-hover cursor-pointer" : "";
  const fontClasses = font === "mono" ? "font-mono" : "font-sans";
  const combinedClasses = `${baseClasses} ${hoverClasses} ${fontClasses} ${className}`;

  const content = (
    <>
      {icon && (
        <div
          className={`${iconBg} ml-[-2px] flex size-5 flex-shrink-0 items-center justify-center rounded-sm`}
        >
          {icon}
        </div>
      )}
      <span className="text-fg-primary truncate">{label}</span>
      {secondaryLabel && (
        <span className="text-fg-tertiary truncate pl-0.5">
          {secondaryLabel}
        </span>
      )}
    </>
  );

  const chipContent = link ? (
    <Link to={link} className={combinedClasses}>
      {content}
    </Link>
  ) : (
    <div className={combinedClasses}>{content}</div>
  );

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>{chipContent}</TooltipTrigger>
          <TooltipContent className="border-border bg-bg-secondary text-fg-primary border shadow-lg">
            {tooltip}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return chipContent;
};

export default Chip;
