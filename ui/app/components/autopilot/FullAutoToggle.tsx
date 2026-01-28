import { useEffect, useState } from "react";
import { Switch, SwitchSize } from "~/components/ui/switch";
import { cn } from "~/utils/common";

const STORAGE_KEY = "autopilot-full-auto";

interface FullAutoToggleProps {
  onCheckedChange?: (checked: boolean) => void;
}

export function FullAutoToggle({ onCheckedChange }: FullAutoToggleProps) {
  const [checked, setChecked] = useState(false);

  // Hydrate from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "true") {
      setChecked(true);
      onCheckedChange?.(true);
    }
  }, [onCheckedChange]);

  const handleChange = (newChecked: boolean) => {
    setChecked(newChecked);
    localStorage.setItem(STORAGE_KEY, String(newChecked));
    onCheckedChange?.(newChecked);
  };

  return (
    <label className="flex cursor-pointer items-center gap-2 select-none">
      <span
        className={cn(
          "text-xs font-medium",
          checked ? "text-orange-600" : "text-fg-tertiary",
        )}
      >
        Full Auto
      </span>
      <Switch
        checked={checked}
        onCheckedChange={handleChange}
        size={SwitchSize.Small}
      />
    </label>
  );
}
