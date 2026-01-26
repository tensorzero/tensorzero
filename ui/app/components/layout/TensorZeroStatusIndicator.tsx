import { useTensorZeroStatusFetcher } from "~/routes/api/tensorzero/status";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { useMemo } from "react";

interface TensorZeroStatusIndicatorProps {
  collapsed?: boolean;
}

/**
 * A component that displays the status of the TensorZero Gateway.
 */
export default function TensorZeroStatusIndicator({
  collapsed = false,
}: TensorZeroStatusIndicatorProps) {
  const { status, isLoading } = useTensorZeroStatusFetcher();
  const uiVersion = __APP_VERSION__;

  // Extract server version from status if available
  const serverVersion = status?.version || "";

  // Check if versions match (ignoring patch version) - memoized to prevent re-renders
  const versionsMatch = useMemo(() => {
    if (!serverVersion) return true; // No data yet, don't show warning
    // We can do an exact match for now
    return uiVersion === serverVersion;
  }, [uiVersion, serverVersion]);

  const statusColor = useMemo(() => {
    if (isLoading || status === undefined) return "bg-gray-300"; // Loading or initial state
    if (!status) return "bg-red-500"; // Could not connect (explicit null/failed state)
    if (!versionsMatch) return "bg-yellow-500"; // Version mismatch
    return "bg-green-500"; // Everything is good
  }, [isLoading, status, versionsMatch]);

  const statusText = isLoading
    ? "Checking status..."
    : status === undefined
      ? "Connecting to Gateway..."
      : status
        ? `TensorZero Gateway ${serverVersion}`
        : "Gateway Unavailable";

  const statusDot = (
    <div className="flex h-4 w-4 shrink-0 items-center justify-center">
      <div className={`h-2 w-2 rounded-full ${statusColor}`} />
    </div>
  );

  const content = (
    <div className="text-fg-muted flex h-8 items-center gap-2 overflow-hidden p-2 text-xs">
      {statusDot}
      <span className="whitespace-nowrap transition-opacity duration-200 group-data-[collapsible=icon]:opacity-0">
        {statusText}
      </span>
      {status && !versionsMatch && !collapsed && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-[10px] text-yellow-600 whitespace-nowrap transition-opacity duration-200 group-data-[collapsible=icon]:opacity-0">
              Version mismatch: UI {uiVersion}
            </span>
          </TooltipTrigger>
          <TooltipContent side="right" align="center">
            Please make sure your UI has the same version as the gateway.
            Otherwise you might have compatibility issues.
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );

  if (collapsed) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent side="right" align="center">
          {statusText}
          {status && !versionsMatch && (
            <div className="text-yellow-600">
              Version mismatch: UI {uiVersion}. Please make sure your UI has the
              same version as the gateway.
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  return content;
}
