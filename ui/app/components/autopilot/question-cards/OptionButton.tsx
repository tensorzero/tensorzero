import { cn } from "~/utils/common";

type OptionButtonProps = {
  isSelected: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
};

export function OptionButton({
  isSelected,
  disabled,
  onClick,
  children,
  className,
}: OptionButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "cursor-pointer rounded-lg border px-3 py-2 text-left transition-all disabled:cursor-not-allowed disabled:opacity-50",
        isSelected
          ? "border-orange-500 bg-orange-50 ring-1 ring-orange-500 ring-inset dark:border-orange-400 dark:bg-orange-950/40 dark:ring-orange-400"
          : "border-border bg-bg-primary hover:border-orange-300 hover:bg-orange-50/50 dark:hover:border-orange-600 dark:hover:bg-orange-950/20",
        className,
      )}
    >
      {children}
    </button>
  );
}
