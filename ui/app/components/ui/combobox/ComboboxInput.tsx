import { forwardRef } from "react";
import { Input } from "~/components/ui/input";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";

type ComboboxInputProps = {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onClick: () => void;
  onBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  placeholder: string;
  disabled?: boolean;
  open: boolean;
  prefix?: React.ReactNode;
  ariaLabel?: string;
};

export const ComboboxInput = forwardRef<HTMLDivElement, ComboboxInputProps>(
  function ComboboxInput(
    {
      value,
      onChange,
      onKeyDown,
      onClick,
      onBlur,
      placeholder,
      disabled = false,
      open,
      prefix,
      ariaLabel,
    },
    ref,
  ) {
    return (
      <div ref={ref} className="relative">
        {prefix && (
          <div className="pointer-events-none absolute inset-y-0 left-3 flex items-center gap-2">
            {prefix}
          </div>
        )}
        <Input
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onClick={onClick}
          onBlur={onBlur}
          placeholder={placeholder}
          disabled={disabled}
          aria-expanded={open}
          aria-label={ariaLabel}
          className={clsx(
            "cursor-text pr-8 font-mono",
            prefix ? "pl-9" : "pl-3",
          )}
        />
        <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center">
          <ChevronDown
            className={clsx("text-fg-tertiary h-4 w-4", open && "-rotate-180")}
          />
        </div>
      </div>
    );
  },
);
