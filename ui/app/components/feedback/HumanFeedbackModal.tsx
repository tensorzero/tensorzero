import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
interface HumanFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HumanFeedbackModal({
  isOpen,
  onClose,
}: HumanFeedbackModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        <div className="mt-4 max-h-[70vh] overflow-y-auto">Sup bb</div>
      </DialogContent>
    </Dialog>
  );
}
