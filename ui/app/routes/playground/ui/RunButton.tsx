import { PlayIcon, Loader2 } from "lucide-react";
import { Button } from "~/components/ui/button";

interface RunButtonProps extends React.PropsWithChildren {
  onRun?: () => void;
  isLoading?: boolean;
  isDisabled?: boolean;
}

export default function RunButton({
  isLoading = false,
  isDisabled = false,
  onRun,
  children,
}: RunButtonProps) {
  return (
    <Button
      size="sm"
      variant={isDisabled ? "ghost" : "default"}
      aria-disabled={isDisabled}
      onClick={onRun}
      disabled={isLoading}
    >
      {children ?? "Run"}
      {isLoading ? (
        <Loader2 className="h-3 w-3 animate-spin" />
      ) : (
        <PlayIcon className="h-3 w-3" />
      )}
    </Button>
  );
}
