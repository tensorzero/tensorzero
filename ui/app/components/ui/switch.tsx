import { cn } from "~/utils/common";

export enum SwitchSize {
  Small = "sm",
  Medium = "md",
}

const sizeStyles: Record<
  SwitchSize,
  { track: string; thumb: string; translate: string }
> = {
  [SwitchSize.Small]: {
    track: "h-4 w-7",
    thumb: "top-0.5 left-0.5 h-3 w-3",
    translate: "translate-x-3",
  },
  [SwitchSize.Medium]: {
    track: "h-5 w-9",
    thumb: "top-0.5 left-0.5 h-4 w-4",
    translate: "translate-x-4",
  },
};

interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  size?: `${SwitchSize}`;
  className?: string;
  id?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
}

export function Switch({
  checked,
  onCheckedChange,
  disabled = false,
  size = SwitchSize.Medium,
  className,
  id,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
}: SwitchProps) {
  const styles = sizeStyles[size];

  return (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={checked}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      disabled={disabled}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        "relative inline-flex shrink-0 cursor-pointer rounded-full transition-colors duration-200",
        "focus-visible:ring-offset-background focus-visible:ring-2 focus-visible:ring-orange-300 focus-visible:ring-offset-2 focus-visible:outline-none",
        "disabled:cursor-not-allowed disabled:opacity-50",
        styles.track,
        checked ? "bg-orange-600" : "bg-bg-muted",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute rounded-full bg-white shadow-sm transition-transform duration-200",
          styles.thumb,
          checked && styles.translate,
        )}
      />
    </button>
  );
}
