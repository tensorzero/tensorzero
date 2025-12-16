import { forwardRef } from "react";
import { Input } from "~/components/ui/input";
import { ChevronDown, X } from "lucide-react";
import clsx from "clsx";
import type { IconProps } from "~/components/icons/Icons";

type IconComponent = React.FC<IconProps>;

type ComboboxInputProps = {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onClick: () => void;
  onBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  placeholder: string;
  disabled?: boolean;
  icon: IconComponent;
  clearable?: boolean;
  onClear?: () => void;
  annotation?: React.ReactNode;
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
      icon: Icon,
      clearable = false,
      onClear,
      annotation,
    },
    ref,
  ) {
    return (
      <div ref={ref} className="relative">
        <div className="pointer-events-none absolute inset-y-0 left-3 flex items-center">
          <Icon
            className={clsx(
              "h-4 w-4",
              value ? "text-fg-primary" : "text-fg-tertiary",
            )}
          />
        </div>
        <Input
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onClick={onClick}
          onBlur={onBlur}
          placeholder={placeholder}
          disabled={disabled}
          className={clsx(
            "cursor-text pl-9 font-mono",
            clearable ? "pr-14" : "pr-8",
          )}
        />
        {annotation && value && (
          <div className="pointer-events-none absolute inset-y-0 left-9 flex items-center">
            <span className="invisible font-mono text-sm">{value}</span>
            <span className="ml-1">{annotation}</span>
          </div>
        )}
        <div className="absolute inset-y-0 right-3 flex items-center gap-1">
          {clearable && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClear?.();
              }}
              className="text-fg-tertiary hover:text-fg-primary cursor-pointer rounded p-0.5"
              aria-label="Clear selection"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
          <ChevronDown className="text-fg-tertiary h-4 w-4" />
        </div>
      </div>
    );
  },
);
