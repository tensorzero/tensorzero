import { Download, Loader2 } from "lucide-react";
import { useFetcher } from "react-router";
import { useEffect, useRef } from "react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { useToast } from "~/hooks/use-toast";

/**
 * Result of applying a config change to file.
 */
interface ApplyConfigResult {
  /** The event ID that was processed */
  eventId: string;
  /** Paths of files that were written */
  writtenPaths: string[];
}

type ApplyAllConfigsResponse =
  | {
      success: true;
      results: ApplyConfigResult[];
      total_processed: number;
    }
  | { success: false; error: string };

interface ApplySessionConfigChangesButtonProps {
  sessionId: string;
  disabled?: boolean;
}

export function ApplySessionConfigChangesButton({
  sessionId,
  disabled,
}: ApplySessionConfigChangesButtonProps) {
  const fetcher = useFetcher<ApplyAllConfigsResponse>();
  const { toast } = useToast();
  const hasShownToastRef = useRef(false);

  const isLoading = fetcher.state !== "idle";

  // Show toast when result arrives
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && !hasShownToastRef.current) {
      hasShownToastRef.current = true;
      if (fetcher.data.success) {
        const totalFiles = fetcher.data.results.reduce(
          (sum, r) => sum + r.writtenPaths.length,
          0,
        );
        toast.success({
          title: "Applied changes to the local filesystem",
          description: `Updated ${totalFiles} ${totalFiles === 1 ? "file" : "files"} based on ${fetcher.data.total_processed} configuration ${fetcher.data.total_processed === 1 ? "change" : "changes"}.`,
        });
      } else {
        toast.error({
          title: "Failed to apply changes to the local filesystem",
          description: fetcher.data.error,
        });
      }
    }

    // Reset the ref when a new submission starts
    if (fetcher.state === "submitting") {
      hasShownToastRef.current = false;
    }
  }, [fetcher.state, fetcher.data, toast]);

  const handleClick = () => {
    fetcher.submit(
      {},
      {
        method: "POST",
        action: `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/config-apply/apply-all`,
        encType: "application/json",
      },
    );
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="outline"
          size="xs"
          onClick={handleClick}
          disabled={disabled || isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Writing...
            </>
          ) : (
            <>
              <Download className="h-4 w-4" />
              Apply changes
            </>
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent
        className="border-border bg-bg-secondary text-fg-primary border text-xs shadow-lg"
        sideOffset={5}
      >
        Apply all configuration changes from this session to the local
        filesystem.
      </TooltipContent>
    </Tooltip>
  );
}
