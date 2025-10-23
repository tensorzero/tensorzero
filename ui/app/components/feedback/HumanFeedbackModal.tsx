import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { useReadOnly } from "~/context/read-only";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

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

  const dialogContent = (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      {!!trigger && (
        <DialogTrigger asChild disabled={isReadOnly}>
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

  // Wrap the trigger with tooltip when in read-only mode
  if (isReadOnly && trigger) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <span className="inline-block">{dialogContent}</span>
          </TooltipTrigger>
          <TooltipContent>
            <p>This feature is not available in read-only mode</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return dialogContent;
}
