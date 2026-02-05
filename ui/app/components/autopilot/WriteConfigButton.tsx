import { Download, Loader2 } from "lucide-react";
import { useFetcher } from "react-router";
import { useEffect, useRef } from "react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { GatewayEvent } from "~/types/tensorzero";
import { useToast } from "~/hooks/use-toast";

type WriteConfigResponse =
  | { success: true; written_paths: string[] }
  | { success: false; error: string };

interface WriteConfigButtonProps {
  sessionId: string;
  event: GatewayEvent;
  disabled?: boolean;
}

export function WriteConfigButton({
  sessionId,
  event,
  disabled,
}: WriteConfigButtonProps) {
  const fetcher = useFetcher<WriteConfigResponse>();
  const { toast } = useToast();
  const hasShownToastRef = useRef(false);

  const isLoading = fetcher.state !== "idle";

  // Show toast when result arrives
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && !hasShownToastRef.current) {
      hasShownToastRef.current = true;
      if (fetcher.data.success) {
        toast.success({
          title: "Config written",
          description: `Wrote ${fetcher.data.written_paths.length} file(s) to disk`,
        });
      } else {
        toast.error({
          title: "Failed to write config",
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
      { event: JSON.stringify(event) },
      {
        method: "POST",
        action: `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/config-writes/write`,
        encType: "application/json",
      },
    );
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="iconSm"
          onClick={handleClick}
          disabled={disabled || isLoading}
          className="h-6 w-6"
          aria-label="Write config to file"
        >
          {isLoading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Download className="h-3.5 w-3.5" />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent
        className="border-border bg-bg-secondary text-fg-primary border text-xs shadow-lg"
        sideOffset={5}
      >
        Write config to file
      </TooltipContent>
    </Tooltip>
  );
}
