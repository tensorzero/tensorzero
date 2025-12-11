import { useCallback, useRef, useState } from "react";

const COMMAND_MENU_KEYS = ["ArrowDown", "ArrowUp", "Enter"];
const RADIX_POPPER_SELECTOR = "[data-radix-popper-content-wrapper]";
const RADIX_SELECT_SELECTOR = "[data-radix-select-content]";

export interface UseComboboxOptions {
  onClose?: () => void;
  openOnFocus?: boolean;
}

export interface UseComboboxReturn {
  open: boolean;
  setOpen: (open: boolean) => void;
  searchValue: string;
  setSearchValue: (value: string) => void;
  isEditing: boolean;
  commandRef: React.RefObject<HTMLDivElement | null>;
  inputValue: (selected: string | null | undefined) => string;
  closeDropdown: () => void;
  handleKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  handleClick: () => void;
  handleFocus: () => void;
}

export function useCombobox(
  options: UseComboboxOptions = {},
): UseComboboxReturn {
  const { onClose, openOnFocus = false } = options;
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const commandRef = useRef<HTMLDivElement | null>(null);

  const closeDropdown = useCallback(() => {
    setOpen(false);
    setSearchValue("");
    setIsEditing(false);
    onClose?.();
  }, [onClose]);

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
      if (
        relatedTarget?.closest(RADIX_POPPER_SELECTOR) ||
        relatedTarget?.closest(RADIX_SELECT_SELECTOR)
      ) {
        return;
      }
      closeDropdown();
    },
    [closeDropdown],
  );

  const handleClick = useCallback(() => {
    setOpen(true);
  }, []);

  const handleFocus = useCallback(() => {
    if (openOnFocus) {
      setOpen(true);
    }
  }, [openOnFocus]);

  const inputValue = useCallback(
    (selected: string | null | undefined) => {
      return isEditing ? searchValue : selected || "";
    },
    [isEditing, searchValue],
  );

  return {
    open,
    setOpen,
    searchValue,
    setSearchValue,
    isEditing,
    commandRef,
    inputValue,
    closeDropdown,
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
    handleFocus,
  };
}
