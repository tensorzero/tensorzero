import { Dialog as DialogPrimitive } from "radix-ui";
import { X } from "lucide-react";
import {
  Dialog,
  DialogContentBox,
  DialogOverlay,
  DialogPortal,
} from "~/components/ui/dialog";
import { Button } from "~/components/ui/button";
import { cn } from "~/utils/common";

interface ErrorDialogProps {
  children: React.ReactNode;
  open: boolean;
  onDismiss: () => void;
  onReopen: () => void;
  label?: string;
}

export function ErrorDialog({
  children,
  open,
  onDismiss,
  onReopen,
  label = "Error",
}: ErrorDialogProps) {
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
            <DialogContentBox className="max-h-[90vh] w-fit gap-0 overflow-hidden rounded-lg p-0">
              {children}

              <DialogPrimitive.Close
                className={cn(
                  "absolute top-4 right-4",
                  "cursor-pointer rounded-sm",
                  "text-muted-foreground opacity-70",
                  "hover:text-foreground transition-opacity hover:opacity-100",
                  "focus:ring-ring focus:ring-offset-background focus:ring-2 focus:ring-offset-2 focus:outline-hidden",
                  "disabled:pointer-events-none",
                )}
              >
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </DialogPrimitive.Close>
            </DialogContentBox>
          </DialogPrimitive.Content>
        </DialogPortal>
      </Dialog>

      {!open && (
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
