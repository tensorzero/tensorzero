import { Dialog as DialogPrimitive } from "radix-ui";
import { X } from "lucide-react";
import {
  Dialog,
  DialogContentBox,
  DialogOverlay,
  DialogPortal,
} from "~/components/ui/dialog";
import { Button } from "~/components/ui/button";

interface ErrorDialogProps {
  children: React.ReactNode;
  open: boolean;
  onDismiss: () => void;
  onReopen: () => void;
  label?: string;
}

/**
 * Dialog for dismissible error states.
 * Reuses Dialog primitives with persistent error indicator when dismissed.
 * Uses dark theme styling to contrast with the dark overlay.
 */
export function ErrorDialog({
  children,
  open,
  onDismiss,
  onReopen,
  label = "Error",
}: ErrorDialogProps) {
  const dismissed = !open;

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => !o && onDismiss()}>
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content
            asChild
            aria-describedby={undefined}
            onOpenAutoFocus={(e) => e.preventDefault()}
          >
            <DialogContentBox className="dark max-h-[90vh] gap-0 bg-neutral-950 p-0 text-neutral-100">
              {children}

              <DialogPrimitive.Close className="absolute top-4 right-4 cursor-pointer rounded-sm text-neutral-400 opacity-70 transition-opacity hover:text-neutral-100 hover:opacity-100 focus:ring-2 focus:ring-neutral-500 focus:ring-offset-2 focus:ring-offset-neutral-950 focus:outline-hidden disabled:pointer-events-none">
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </DialogPrimitive.Close>
            </DialogContentBox>
          </DialogPrimitive.Content>
        </DialogPortal>
      </Dialog>

      {dismissed && (
        <Button
          onClick={onReopen}
          variant="destructive"
          className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 rounded-full shadow-lg"
          slotLeft={<span className="h-2 w-2 rounded-full bg-white" />}
        >
          {label}
        </Button>
      )}
    </>
  );
}
