import React from "react";
import { Link } from "react-router";
import {
  Functions,
  Episodes,
  SupervisedFineTuning,
  Placeholder,
} from "~/components/icons/Icons";

interface ChipProps {
  text: string;
  className?: string;
}

export const Chip: React.FC<ChipProps> = ({ text, className = "" }) => {
  return (
    <div
      className={`inline-flex items-center rounded-md bg-background-tertiary px-2 py-1 text-sm text-foreground-primary ${className}`}
    >
      {text}
    </div>
  );
};

// Episode Chip

interface EpisodeChipProps extends ChipProps {
  link: string;
}

export const EpisodeChip: React.FC<EpisodeChipProps> = ({
  text,
  link,
  className = "",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md font-mono hover:bg-background-muted";

  return (
    <Link to={link} className={`${baseClasses} cursor-pointer ${className}`}>
      <Episodes className="text-foreground-tertiary" size={16} />
      <span>{text}</span>
    </Link>
  );
};

// Function Chip

interface FunctionChipProps {
  name: string;
  link: string;
  type: string;
  className?: string;
}

export const FunctionChip: React.FC<FunctionChipProps> = ({
  name,
  link,
  type,
  className = "",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md font-mono hover:bg-background-muted";

  return (
    <Link to={link} className={`${baseClasses} cursor-pointer ${className}`}>
      <Functions size={16} />
      <span>{name}</span>
      <span className="text-foreground-tertiary">{type}</span>
    </Link>
  );
};

// Variant Chip

interface VariantChipProps {
  name: string;
  link: string;
  type?: string;
  className?: string;
}

export const VariantChip: React.FC<VariantChipProps> = ({
  name,
  link,
  type,
  className = "",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md font-mono hover:bg-background-muted";

  return (
    <Link to={link} className={`${baseClasses} cursor-pointer ${className}`}>
      <SupervisedFineTuning size={16} />
      <span>{name}</span>
      {type && <span className="text-foreground-tertiary">{type}</span>}
    </Link>
  );
};

// Timestamp Chip

interface TimestampChipProps {
  timestamp: string | number | Date;
  className?: string;
}

export const TimestampChip: React.FC<TimestampChipProps> = ({
  timestamp,
  className = "",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md";

  const formatDate = (date: Date) => {
    const options: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      hour12: true,
    };

    return new Date(date).toLocaleString("en-US", options);
  };

  const formattedDate = formatDate(new Date(timestamp));

  return (
    <div className={`${baseClasses} ${className}`}>
      <Placeholder className="text-foreground-tertiary" size={16} />
      <span>{formattedDate}</span>
    </div>
  );
};

// Processing Time Chip

interface ProcessingTimeChipProps {
  processingTimeMs: number;
  className?: string;
}

export const ProcessingTimeChip: React.FC<ProcessingTimeChipProps> = ({
  processingTimeMs,
  className = "",
}) => {
  const baseClasses =
    "inline-flex items-center text-sm text-foreground-primary py-1 px-2 gap-2 rounded-md";

  return (
    <div className={`${baseClasses} ${className}`}>
      <Placeholder className="text-foreground-tertiary" size={16} />
      <span>{processingTimeMs}ms</span>
    </div>
  );
};

export default Chip;
