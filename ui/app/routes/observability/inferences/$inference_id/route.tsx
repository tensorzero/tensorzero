import {
  parseInferenceOutput,
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference";
import {
  queryDemonstrationFeedbackByInferenceId,
  queryFeedbackBoundsByTargetId,
  queryFeedbackByTargetId,
} from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  redirect,
  useFetcher,
  useNavigate,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import BasicInfo from "./BasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "./InferenceParameters";
import { TagsTable } from "~/components/utils/TagsTable";
import { ModelInferencesAccordion } from "./ModelInferencesAccordion";
import { useState } from "react";
import { useConfig } from "~/context/config";
import { VariantResponseModal } from "./VariantResponseModal";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import {
  getDatasetCounts,
  insertDatapoint,
} from "~/utils/clickhouse/datasets.server";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import { inferenceRowToDatasetRow } from "~/utils/clickhouse/datasets";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const [
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    demonstration_feedback,
  ] = await Promise.all([
    queryInferenceById(inference_id),
    queryModelInferencesByInferenceId(inference_id),
    queryFeedbackByTargetId({
      target_id: inference_id,
      before: beforeFeedback || undefined,
      after: afterFeedback || undefined,
      page_size: pageSize,
    }),
    queryFeedbackBoundsByTargetId({ target_id: inference_id }),
    getDatasetCounts(),
    queryDemonstrationFeedbackByInferenceId({
      inference_id,
      page_size: 1,
    }),
  ]);
  if (!inference) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }

  return {
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    hasDemonstration: demonstration_feedback.length > 0,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const dataset = formData.get("dataset");
  const output = formData.get("output");
  const inference_id = formData.get("inference_id");
  if (!dataset || !output || !inference_id) {
    throw data("Missing required fields", { status: 400 });
  }
  const promises = [queryInferenceById(inference_id.toString())] as const;
  let datapoint: ParsedDatasetRow;
  if (output === "demonstration") {
    const [inference, demonstration_feedback] = await Promise.all([
      ...promises,
      queryDemonstrationFeedbackByInferenceId({
        inference_id: inference_id.toString(),
        page_size: 1,
      }),
    ]);
    if (!inference) {
      throw data("No inference found", { status: 404 });
    }
    datapoint = inferenceRowToDatasetRow(inference, dataset.toString());
    datapoint.output = parseInferenceOutput(demonstration_feedback[0].value);
  } else {
    const [inference] = await Promise.all(promises);
    if (!inference) {
      throw data("No inference found", { status: 404 });
    }
    datapoint = inferenceRowToDatasetRow(inference, dataset.toString());
    if (output === "none") {
      datapoint.output = undefined;
    }
  }
  await insertDatapoint(datapoint);
  return redirect(`/datasets/${dataset.toString()}/datapoint/${datapoint.id}`);
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    hasDemonstration,
  } = loaderData;
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [variantInferenceIsLoading, setVariantInferenceIsLoading] =
    useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  const topFeedback = feedback[0] as { id: string } | undefined;
  const bottomFeedback = feedback[feedback.length - 1] as
    | { id: string }
    | undefined;

  const handleNextFeedbackPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousFeedbackPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  // These are swapped because the table is sorted in descending order
  const disablePreviousFeedbackPage =
    !topFeedback?.id ||
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNextFeedbackPage =
    !bottomFeedback?.id ||
    !feedback_bounds.first_id ||
    feedback_bounds.first_id === bottomFeedback.id;

  const num_feedbacks = feedback.length;

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setVariantInferenceIsLoading(false);
  };
  const config = useConfig();
  const variants = Object.keys(
    config.functions[inference.function_name]?.variants || {},
  );
  const addToDatasetFetcher = useFetcher();

  const handleAddToDataset = (
    dataset: string,
    output: "inference" | "demonstration" | "none",
  ) => {
    const formData = new FormData();
    formData.append("dataset", dataset);
    formData.append("output", output);
    formData.append("inference_id", inference.id);
    addToDatasetFetcher.submit(formData, { method: "post", action: "." });
  };

  return (
    <div className="container mx-auto px-4 pb-8">
      <PageLayout>
        <PageHeader heading="Inference" name={inference.id} />

        <SectionsGroup>
          <SectionLayout>
            <BasicInfo
              inference={inference}
              inferenceUsage={getTotalInferenceUsage(model_inferences)}
              tryWithVariantProps={{
                variants,
                onVariantSelect,
                isLoading: variantInferenceIsLoading,
              }}
              dataset_counts={dataset_counts}
              onDatasetSelect={handleAddToDataset}
              hasDemonstration={hasDemonstration}
            />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Input" />
            <Input input={inference.input} />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Output" />
            <Output output={inference.output} />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader
              heading="Feedback"
              count={num_feedbacks}
              badge={{
                name: "inference",
                tooltip:
                  "This table only includes inference-level feedback. To see episode-level feedback, open the detail page for that episode.",
              }}
            />
            <FeedbackTable feedback={feedback} />
            <PageButtons
              onNextPage={handleNextFeedbackPage}
              onPreviousPage={handlePreviousFeedbackPage}
              disableNext={disableNextFeedbackPage}
              disablePrevious={disablePreviousFeedbackPage}
            />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Inference Parameters" />
            <ParameterCard parameters={inference.inference_params} />
          </SectionLayout>

          {inference.function_type === "chat" && (
            <SectionLayout>
              <SectionHeader heading="Tool Parameters" />
              <ParameterCard parameters={inference.tool_params} />
            </SectionLayout>
          )}

          {inference.function_type === "json" && (
            <SectionLayout>
              <SectionHeader heading="Output Schema" />
              <ParameterCard parameters={inference.output_schema} />
            </SectionLayout>
          )}

          {Object.keys(inference.tags).length > 0 && (
            <SectionLayout>
              <SectionHeader heading="Tags" />
              <TagsTable tags={inference.tags} />
            </SectionLayout>
          )}

          <SectionLayout>
            <SectionHeader heading="Model Inferences" />
            <ModelInferencesAccordion modelInferences={model_inferences} />
          </SectionLayout>
        </SectionsGroup>

        {selectedVariant && (
          <VariantResponseModal
            isOpen={isModalOpen}
            isLoading={variantInferenceIsLoading}
            setIsLoading={setVariantInferenceIsLoading}
            onClose={handleModalClose}
            inference={inference}
            inferenceUsage={getTotalInferenceUsage(model_inferences)}
            selectedVariant={selectedVariant}
          />
        )}
      </PageLayout>
    </div>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
