import type { Control } from "react-hook-form";
import { useWatch } from "react-hook-form";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { SFTFormValues } from "./types";

export function SFTCountDisplay({
  control,
  functionInferenceCount,
  metricFeedbackCount,
  metricCuratedInferenceCount,
  isLoading,
}: {
  control: Control<SFTFormValues>;
  functionInferenceCount: number | null;
  metricFeedbackCount: number | null;
  metricCuratedInferenceCount: number | null;
  isLoading: boolean;
}) {
  const [showSplit, setShowSplit] = useState(false);

  const watchedFields = useWatch({
    control,
    name: ["maxSamples", "validationSplitPercent"] as const,
  });

  const [maxSamples, validationSplitPercent] = watchedFields;

  if (isLoading) {
    return <div className="text-fg-secondary text-sm">Loading counts...</div>;
  }

  // Calculate training and validation split counts
  const availableForTraining =
    metricCuratedInferenceCount ?? functionInferenceCount;
  const actualSamples = availableForTraining
    ? Math.min(maxSamples || 0, availableForTraining)
    : 0;
  const validationCount = Math.floor(
    (actualSamples * (validationSplitPercent || 0)) / 100,
  );
  const trainingCount = actualSamples - validationCount;

  // Check if curated inference count is too low
  const isCuratedInferenceCountLow =
    metricCuratedInferenceCount !== null && metricCuratedInferenceCount < 10;

  // Determine if any count is available to show
  const hasAnyCountToShow =
    functionInferenceCount !== null ||
    metricFeedbackCount !== null ||
    metricCuratedInferenceCount !== null;

  if (!hasAnyCountToShow) {
    return (
      <div className="text-fg-secondary text-sm">
        <div className="flex flex-row items-center justify-between gap-2 pt-2">
          <span>Select a function and metric to see training data counts.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="text-fg-secondary w-full text-sm">
      {functionInferenceCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Function Inferences</span>
          <span className="font-medium">
            {functionInferenceCount.toLocaleString()}
          </span>
        </div>
      )}

      {metricFeedbackCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Metric Feedbacks</span>
          <span className="font-medium">
            {metricFeedbackCount.toLocaleString()}
          </span>
        </div>
      )}

      {metricCuratedInferenceCount !== null && (
        <div className="border-border flex flex-row items-center justify-between gap-2 border-b py-2">
          <span>Curated Inferences</span>
          <div className="flex items-center gap-2">
            <span className="font-medium">
              {metricCuratedInferenceCount.toLocaleString()}
            </span>
            {isCuratedInferenceCountLow && (
              <span className="rounded bg-yellow-50 px-2 py-1 text-xs text-yellow-600">
                Low count
              </span>
            )}
          </div>
        </div>
      )}

      {actualSamples > 0 && (
        <>
          <div className="flex flex-row items-center justify-between gap-2 py-2">
            <div className="flex items-center gap-2">
              <span className="text-fg-primary font-medium">Total Samples</span>
              <button
                type="button"
                onClick={() => setShowSplit(!showSplit)}
                className="text-fg-secondary flex items-center gap-1 rounded-md bg-neutral-200 py-px pr-1 pl-1.5 text-xs font-medium transition-colors hover:bg-neutral-300"
              >
                <span>Show Split</span>
                <ChevronDown
                  className={`h-3 w-3 transition-transform ${showSplit ? "rotate-180" : ""}`}
                />
              </button>
            </div>
            <span className="text-fg-primary font-semibold">
              {actualSamples.toLocaleString()}
            </span>
          </div>

          {showSplit && (
            <>
              <div className="border-border flex flex-row items-center justify-between gap-2 border-t py-2">
                <span>Training Samples</span>
                <span className="font-medium">
                  {trainingCount.toLocaleString()}
                </span>
              </div>

              <div className="border-border flex flex-row items-center justify-between gap-2 border-t pt-2">
                <span>Validation Samples</span>
                <span className="font-medium">
                  {validationCount.toLocaleString()}
                </span>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
