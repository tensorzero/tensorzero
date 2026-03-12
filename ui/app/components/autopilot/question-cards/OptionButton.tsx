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
          ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
          : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
        className,
      )}
    >
      {children}
    </button>
  );
}
