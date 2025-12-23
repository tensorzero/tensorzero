import { useState, useCallback } from "react";
import { useToast } from "~/hooks/use-toast";

export function useCopy(resetDelayMs = 2000): {
  copy: (text: string) => Promise<void>;
  didCopy: boolean;
  isCopyAvailable: boolean;
} {
  const [didCopy, setDidCopy] = useState<boolean>(false);
  const { toast } = useToast();

  const copy = useCallback(
    async (text: string) => {
      if (!text) return;

      try {
        await navigator.clipboard.writeText(text);
        setDidCopy(true);

        // Reset the copy state after the specified delay
        setTimeout(() => setDidCopy(false), resetDelayMs);
      } catch {
        toast.error({ title: "Failed to copy", log: true });
      }
    },
    [toast, resetDelayMs],
  );

  const isSecureContext =
    typeof window !== "undefined" && window.isSecureContext;
  const isClipboard = typeof navigator !== "undefined" && !!navigator.clipboard;
  const isCopyAvailable = isSecureContext && isClipboard;

  return { copy, didCopy, isCopyAvailable };
}
