import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";

interface HumanFeedbackModalProps {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
  trigger?: React.ReactElement;
  children?: React.ReactNode;
  disabled?: boolean;
}

export function HumanFeedbackModal({
  isOpen,
  onOpenChange,
  trigger,
  children,
  disabled = false,
}: HumanFeedbackModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      {!!trigger && (
        <DialogTrigger asChild disabled={disabled}>
          {trigger}
        </DialogTrigger>
      )}
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        <DialogBody>{children}</DialogBody>
      </DialogContent>
    </Dialog>
  );
}
