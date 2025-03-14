import React from "react";
import { Link } from "react-router";

interface ChipProps {
  label: string;
  icon?: React.ReactNode;
  secondaryLabel?: string;
  link?: string;
  className?: string;
  font?: "sans" | "mono";
}

export const Chip: React.FC<ChipProps> = ({
  label,
  icon,
  secondaryLabel,
  link,
  className = "",
  font = "sans",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md";
  const hoverClasses = link ? "hover:bg-background-muted cursor-pointer" : "";
  const fontClasses = font === "mono" ? "font-mono" : "font-sans";
  const combinedClasses = `${baseClasses} ${hoverClasses} ${fontClasses} ${className}`;

  const content = (
    <>
      {icon}
      <span>{label}</span>
      {secondaryLabel && (
        <span className="text-foreground-tertiary">{secondaryLabel}</span>
      )}
    </>
  );

  if (link) {
    return (
      <Link to={link} className={combinedClasses}>
        {content}
      </Link>
    );
  }

  return <div className={combinedClasses}>{content}</div>;
};

export default Chip;
