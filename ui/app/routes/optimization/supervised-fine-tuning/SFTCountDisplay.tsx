import type { Control } from "react-hook-form";
import { useWatch } from "react-hook-form";
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
  const watchedFields = useWatch({
    control,
    name: ["maxSamples", "validationSplitPercent"] as const,
  });

  const [maxSamples, validationSplitPercent] = watchedFields;

  if (isLoading) {
    return <div className="text-sm text-fg-secondary">Loading counts...</div>;
  }

  // Calculate training and validation split counts
  const availableForTraining = metricCuratedInferenceCount ?? functionInferenceCount;
  const actualSamples = availableForTraining ? Math.min(maxSamples || 0, availableForTraining) : 0;
  const validationCount = Math.floor((actualSamples * (validationSplitPercent || 0)) / 100);
  const trainingCount = actualSamples - validationCount;

  // Check if curated inference count is too low
  const isCuratedInferenceCountLow = metricCuratedInferenceCount !== null && metricCuratedInferenceCount < 10;

  // Determine if any count is available to show
  const hasAnyCountToShow = 
    functionInferenceCount !== null || 
    metricFeedbackCount !== null || 
    metricCuratedInferenceCount !== null;

  if (!hasAnyCountToShow) {
    return (
      <div className="text-sm text-fg-secondary">
        <div className="flex flex-row items-center pt-2 justify-between gap-2">
          <span>Select a function and metric to see training data counts.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full text-sm text-fg-secondary">
      {functionInferenceCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Function Inferences</span>
          <span className="font-medium">{functionInferenceCount.toLocaleString()}</span>
        </div>
      )}
      
      {metricFeedbackCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Metric Feedbacks</span>
          <span className="font-medium">{metricFeedbackCount.toLocaleString()}</span>
        </div>
      )}
      
      {metricCuratedInferenceCount !== null && (
        <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
          <span>Curated Inferences</span>
          <div className="flex items-center gap-2">
            <span className="font-medium">{metricCuratedInferenceCount.toLocaleString()}</span>
            {isCuratedInferenceCountLow && (
              <span className="text-xs text-yellow-600 bg-yellow-50 px-2 py-1 rounded">
                Low count
              </span>
            )}
          </div>
        </div>
      )}

      {actualSamples > 0 && (
        <>
          <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
            <span>Total Samples</span>
            <span className="font-medium">{actualSamples.toLocaleString()}</span>
          </div>
          
          <div className="flex flex-row items-center py-2 border-b border-border justify-between gap-2">
            <span>Training Samples</span>
            <span className="font-medium">{trainingCount.toLocaleString()}</span>
          </div>
          
          <div className="flex flex-row items-center pt-2 font-medium text-fg-primary justify-between gap-2">
            <span>Validation Samples</span>
            <span className="font-medium">{validationCount.toLocaleString()}</span>
          </div>
        </>
      )}
    </div>
  );
}
