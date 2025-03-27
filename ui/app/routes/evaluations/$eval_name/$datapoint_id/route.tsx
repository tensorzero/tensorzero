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
  return {
    evalResults,
    selected_eval_run_ids_array,
    eval_name,
    datapoint_id,
  };
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const { evalResults, selected_eval_run_ids_array, eval_name, datapoint_id } =
    loaderData;
  return (
    <PageLayout>
      <PageHeader label="Datapoint" name={datapoint_id} />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Input input={evalResults[0].input} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Output" />
          {evalResults.map((result) => (
            <div key={result.eval_run_id}>
              <SectionHeader heading={result.eval_run_id} />
              <Output output={result.generated_output} />
            </div>
          ))}
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
