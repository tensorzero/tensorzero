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
  prominence?: "normal" | "muted";
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
  prominence = "normal",
}) => {
  const baseClasses =
    "inline-flex text-sm text-fg-primary px-0 md:px-2 gap-1.5 rounded-md whitespace-nowrap overflow-hidden";
  const hoverClasses = link ? "md:hover:bg-bg-hover cursor-pointer" : "";
  const fontClasses = font === "mono" ? "font-mono" : "font-sans";
  const combinedClasses = clsx(
    baseClasses,
    hoverClasses,
    fontClasses,
    className,
  );

  const content = (
    <>
      {icon && (
        <div
          className={clsx(
            iconBg,
            "flex size-5 flex-shrink-0 items-center justify-center rounded-sm md:ml-[-2px]",
          )}
        >
          {icon}
        </div>
      )}
      <span
        className={clsx(
          "truncate",
          prominence === "muted" ? "text-fg-muted" : "text-fg-primary",
        )}
      >
        {label}
      </span>
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
