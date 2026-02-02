import { Download, Loader2 } from "lucide-react";
import { useFetcher } from "react-router";
import { useEffect, useRef } from "react";
import { Button } from "~/components/ui/button";
import { useToast } from "~/hooks/use-toast";
import type { WriteConfigWriteResult } from "~/types/tensorzero";

type WriteAllConfigsResponse =
  | {
      success: true;
      results: WriteConfigWriteResult[];
      total_processed: number;
    }
  | { success: false; error: string };

interface WriteAllConfigsButtonProps {
  sessionId: string;
  disabled?: boolean;
}

export function WriteAllConfigsButton({
  sessionId,
  disabled,
}: WriteAllConfigsButtonProps) {
  const fetcher = useFetcher<WriteAllConfigsResponse>();
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
          title: "Configs written",
          description: `Wrote ${totalFiles} file(s) from ${fetcher.data.total_processed} config(s)`,
        });
      } else {
        toast.error({
          title: "Failed to write configs",
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
        action: `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/config-writes/write-all`,
        encType: "application/json",
      },
    );
  };

  return (
    <Button
      variant="outline"
      size="sm"
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
          Write All Configs
        </>
      )}
    </Button>
  );
}
