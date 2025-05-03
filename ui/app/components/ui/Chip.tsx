import React from "react";
import { Link } from "react-router";
import clsx from "clsx";
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

const Chip: React.FC<ChipProps> = ({
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
    "inline-flex text-sm text-fg-primary py-1 px-0 md:px-2 gap-1.5 rounded-md whitespace-nowrap overflow-hidden";
  const hoverClasses = link ? "md:hover:bg-bg-hover cursor-pointer" : "";
  const fontClasses = font === "mono" ? "font-mono" : "font-sans";
  const combinedClasses = clsx(baseClasses, hoverClasses, fontClasses, className);

  const content = (
    <>
      {icon && (
        <div
          className={clsx(iconBg, "md:ml-[-2px] flex size-5 items-center justify-center rounded-sm flex-shrink-0")}
        >
          {icon}
        </div>
      )}
      <span className="text-fg-primary truncate">{label}</span>
      {secondaryLabel && (
        <span className="text-fg-tertiary pl-0.5 truncate">{secondaryLabel}</span>
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

  return tooltip ? (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>{chipContent}</TooltipTrigger>
        <TooltipContent className="border-border bg-bg-secondary text-fg-primary border shadow-lg">
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  ) : (
    chipContent
  );
};

export default Chip;