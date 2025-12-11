import { forwardRef } from "react";
import { Input } from "~/components/ui/input";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import type { IconProps } from "~/components/icons/Icons";

type IconComponent = React.FC<IconProps>;

interface ComboboxInputProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onFocus: () => void;
  onClick: () => void;
  onBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  placeholder: string;
  disabled?: boolean;
  monospace?: boolean;
  open: boolean;
  icon: IconComponent;
  iconClassName?: string;
}

export const ComboboxInput = forwardRef<HTMLDivElement, ComboboxInputProps>(
  function ComboboxInput(
    {
      value,
      onChange,
      onKeyDown,
      onFocus,
      onClick,
      onBlur,
      placeholder,
      disabled = false,
      monospace = false,
      open,
      icon: Icon,
      iconClassName,
    },
    ref,
  ) {
    return (
      <div ref={ref} className="relative">
        <div className="pointer-events-none absolute inset-y-0 left-3 flex items-center">
          <Icon
            className={clsx(
              "h-4 w-4",
              iconClassName ?? (value ? "text-fg-primary" : "text-fg-tertiary"),
            )}
          />
        </div>
        <Input
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onClick={onClick}
          onBlur={onBlur}
          placeholder={placeholder}
          disabled={disabled}
          className={clsx("cursor-text pr-8 pl-9", monospace && "font-mono")}
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
