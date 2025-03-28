import type { DisplayEvalError, EvalError } from "~/utils/evals";
import EvalRunBadge from "~/components/evaluations/EvalRunBadge";

export interface EvalErrorDisplayInfo {
  variantName: string;
  errors: DisplayEvalError[];
}

export interface EvalErrorInfoProps {
  errors: Record<string, EvalErrorDisplayInfo>;
}

export function EvalErrorInfo({ errors }: EvalErrorInfoProps) {
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
          eval_run_id: key,
          variant_name: variantName,
        };

        return (
          <div key={key} className="mt-2">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <EvalRunBadge runInfo={runInfo} getColor={getColor} />
            </h3>
            <div className="mb-2 mt-1 max-h-64 overflow-y-auto rounded border p-2">
              {errors.map((error, index) => (
                <EvalError key={index} error={error} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function EvalError({ error }: { error: DisplayEvalError }) {
  return (
    <div className="font-mono text-xs">
      {error.datapoint_id && <h3>{error.datapoint_id}</h3>}
      <p>{error.message}</p>
    </div>
  );
}
