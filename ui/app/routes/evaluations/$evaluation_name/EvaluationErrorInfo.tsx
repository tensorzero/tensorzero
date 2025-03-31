import type {
  DisplayEvaluationError,
  EvaluationError,
} from "~/utils/evaluations";
import EvaluationRunBadge from "~/components/evaluations/EvaluationRunBadge";

export interface EvaluationErrorDisplayInfo {
  variantName: string;
  errors: DisplayEvaluationError[];
}

export interface EvaluationErrorInfoProps {
  errors: Record<string, EvaluationErrorDisplayInfo>;
}

export function EvaluationErrorInfo({ errors }: EvaluationErrorInfoProps) {
  const sortedErrorEntries = Object.entries(errors).sort((a, b) =>
    b[0].localeCompare(a[0]),
  ); // Sort in reverse order by key
  const getColor = () => {
    return "bg-red-600 hover:bg-red-700";
  };

  return (
    <div>
      {sortedErrorEntries.map(([key, { variantName, errors }]) => {
        if (!errors || errors.length === 0) return null;
        const runInfo = {
          evaluation_run_id: key,
          variant_name: variantName,
        };

        return (
          <div key={key} className="mt-2">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <EvaluationRunBadge runInfo={runInfo} getColor={getColor} />
            </h3>
            <div className="mb-2 mt-1 max-h-64 overflow-y-auto rounded border p-2">
              {errors.map((error, index) => (
                <EvaluationError key={index} error={error} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function EvaluationError({ error }: { error: DisplayEvaluationError }) {
  return (
    <div className="font-mono text-xs">
      {error.datapoint_id && <h3>{error.datapoint_id}</h3>}
      <p>{error.message}</p>
    </div>
  );
}
