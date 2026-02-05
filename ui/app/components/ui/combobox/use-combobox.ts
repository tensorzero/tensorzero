import { useCallback, useRef, useState } from "react";

const COMMAND_MENU_KEYS = ["ArrowDown", "ArrowUp", "Enter"];
const RADIX_POPPER_SELECTOR = "[data-radix-popper-content-wrapper]";

/**
 * Shared state and handlers for combobox behavior.
 *
 * Currently used by the Combobox component. Can be exported in the future for
 * custom combobox implementations that require specialized rendering or behavior
 * beyond what the base component provides.
 */
export function useCombobox() {
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const commandRef = useRef<HTMLDivElement | null>(null);

  const closeDropdown = useCallback(() => {
    setOpen(false);
    setSearchValue("");
    setIsEditing(false);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Escape") {
        closeDropdown();
        return;
      }

      if (COMMAND_MENU_KEYS.includes(e.key)) {
        e.preventDefault();
        if (!open) setOpen(true);
        commandRef.current?.dispatchEvent(
          new KeyboardEvent("keydown", { key: e.key, bubbles: true }),
        );
      }
    },
    [closeDropdown, open],
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearchValue(e.target.value);
      setIsEditing(true);
      if (!open) setOpen(true);
    },
    [open],
  );

  const handleBlur = useCallback(
    (e: React.FocusEvent<HTMLInputElement>) => {
      const relatedTarget = e.relatedTarget as Element | null;
      if (relatedTarget?.closest(RADIX_POPPER_SELECTOR)) {
        return;
      }
      closeDropdown();
    },
    [closeDropdown],
  );

  const handleClick = useCallback(() => {
    setOpen(true);
  }, []);

  const getInputValue = useCallback(
    (selected: string | null) => {
      return isEditing ? searchValue : selected || "";
    },
    [isEditing, searchValue],
  );

  return {
    open,
    searchValue,
    isEditing,
    commandRef,
    getInputValue,
    closeDropdown,
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  };
}
