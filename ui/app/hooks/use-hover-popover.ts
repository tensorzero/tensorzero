import { useState, useRef, useCallback, useEffect, useMemo } from "react";

const CLOSE_DELAY_MS = 150;

interface HoverPopoverState {
  open: boolean;
  setOpen: (open: boolean) => void;
  scheduleClose: () => void;
  cancelClose: () => void;
  triggerProps: {
    onPointerEnter: () => void;
    onPointerLeave: () => void;
  };
  contentProps: {
    onPointerEnter: () => void;
    onPointerLeave: () => void;
  };
}

export function useHoverPopover(): HoverPopoverState {
  const [open, setOpen] = useState(false);
  const closeTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleClose = useCallback(() => {
    closeTimeout.current = setTimeout(() => setOpen(false), CLOSE_DELAY_MS);
  }, []);

  const cancelClose = useCallback(() => {
    if (closeTimeout.current) {
      clearTimeout(closeTimeout.current);
      closeTimeout.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (closeTimeout.current) {
        clearTimeout(closeTimeout.current);
      }
    };
  }, []);

  const onPointerEnter = useCallback(() => {
    cancelClose();
    setOpen(true);
  }, [cancelClose]);

  const triggerProps = useMemo(
    () => ({ onPointerEnter, onPointerLeave: scheduleClose }),
    [onPointerEnter, scheduleClose],
  );

  const contentProps = useMemo(
    () => ({ onPointerEnter: cancelClose, onPointerLeave: scheduleClose }),
    [cancelClose, scheduleClose],
  );

  return {
    open,
    setOpen,
    scheduleClose,
    cancelClose,
    triggerProps,
    contentProps,
  };
}
