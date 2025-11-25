import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

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
  const isReadOnly = useReadOnly();

  return (
    <ReadOnlyGuard asChild>
      <Dialog open={isOpen} onOpenChange={onOpenChange}>
        {!!trigger && (
          <DialogTrigger asChild disabled={isReadOnly}>
            {trigger}
          </DialogTrigger>
        )}
        <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
          <DialogHeader>
            <DialogTitle>Add Feedback</DialogTitle>
            <DialogDescription>
              Assign feedback to improve future responses.
            </DialogDescription>
          </DialogHeader>
          <DialogBody>{children}</DialogBody>
        </DialogContent>
      </Dialog>
    </ReadOnlyGuard>
  );
}
