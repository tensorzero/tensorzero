import { getEvalsForDatapoint } from "~/utils/clickhouse/evaluations.server";
import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import Input from "~/components/inference/Input";
import { redirect } from "react-router";
import Output from "~/components/inference/Output";
import {
  consolidate_eval_results,
  getEvaluatorMetricName,
} from "~/utils/clickhouse/evaluations";
import { useConfig } from "~/context/config";

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const eval_name = params.eval_name;
  const datapoint_id = params.datapoint_id;
  const dataset_name = config.evals[eval_name].dataset_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);

  const selected_eval_run_ids = searchParams.get("eval_run_ids");
  const selected_eval_run_ids_array = selected_eval_run_ids
    ? selected_eval_run_ids.split(",")
    : [];
  if (selected_eval_run_ids_array.length === 0) {
    return redirect(`/datasets/${dataset_name}/datapoint/${datapoint_id}`);
  }
  const evalResults = await getEvalsForDatapoint(
    eval_name,
    datapoint_id,
    selected_eval_run_ids_array,
  );
  const consolidatedEvalResults = consolidate_eval_results(evalResults);
  if (consolidatedEvalResults.length !== selected_eval_run_ids_array.length) {
    // Find which eval run IDs are missing from the results
    const foundEvalRunIds = new Set(
      consolidatedEvalResults.map((result) => result.eval_run_id),
    );
    const missingEvalRunIds = selected_eval_run_ids_array.filter(
      (id) => !foundEvalRunIds.has(id),
    );

    return new Response(
      `Evaluation run ID(s) not found: ${missingEvalRunIds.join(", ")}`,
      { status: 404 },
    );
  }
  return {
    consolidatedEvalResults,
    selected_eval_run_ids_array,
    eval_name,
    datapoint_id,
  };
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    consolidatedEvalResults,
    selected_eval_run_ids_array,
    eval_name,
    datapoint_id,
  } = loaderData;
  loaderData;
  const config = useConfig();
  const eval_config = config.evals[eval_name];
  const metric_name_to_evaluator_config = Object.fromEntries(
    Object.entries(eval_config.evaluators).map(
      ([evaluator_name, evaluator_config]) => [
        getEvaluatorMetricName(eval_name, evaluator_name),
        evaluator_config,
      ],
    ),
  );
  const outputsToDisplay = [
    {
      id: "Reference",
      output: consolidatedEvalResults[0].reference_output,
    },
    ...consolidatedEvalResults.map((result) => ({
      id: result.eval_run_id,
      output: result.generated_output,
    })),
  ];
  console.log(outputsToDisplay);
  return (
    <PageLayout>
      <PageHeader label="Datapoint" name={datapoint_id} />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Input input={consolidatedEvalResults[0].input} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Output" />
          <div className="flex gap-4 overflow-x-auto pb-4">
            {outputsToDisplay.map((result) => (
              <div
                key={result.id}
                className="min-w-[300px] max-w-[450px] flex-shrink-0"
              >
                <div className="mb-2">
                  <button className="rounded-md bg-slate-700 px-4 py-2 text-sm text-white">
                    {result.id}
                  </button>
                </div>
                <Output output={result.output} />
              </div>
            ))}
          </div>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
