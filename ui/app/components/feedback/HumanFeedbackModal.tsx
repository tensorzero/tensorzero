import {
  Dialog,
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
}

export function HumanFeedbackModal({
  isOpen,
  onOpenChange,
  trigger,
  children,
}: HumanFeedbackModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      {!!trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>}
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        {children}
      </DialogContent>
    </Dialog>
  );
}
