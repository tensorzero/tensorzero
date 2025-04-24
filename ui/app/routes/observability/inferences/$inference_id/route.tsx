import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference";
import {
  pollForFeedbackItem,
  queryDemonstrationFeedbackByInferenceId,
  queryFeedbackBoundsByTargetId,
  queryFeedbackByTargetId,
} from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  Link,
  redirect,
  useFetcher,
  useNavigate,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import BasicInfo from "./InferenceBasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/NewOutput";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { addHumanFeedback, tensorZeroClient } from "~/utils/tensorzero.server";
import { ParameterCard } from "./InferenceParameters";
import { TagsTable } from "~/components/utils/TagsTable";
import { ModelInferencesTable } from "./ModelInferencesTable";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useConfig } from "~/context/config";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { InferenceActions } from "./InferenceActions";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";
import { Toaster } from "~/components/ui/toaster";
import { useToast } from "~/hooks/use-toast";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;

  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  // --- Define all promises, conditionally choosing the feedback promise ---

  const inferencePromise = queryInferenceById(inference_id);
  const modelInferencesPromise =
    queryModelInferencesByInferenceId(inference_id);
  const datasetCountsPromise = getDatasetCounts();
  const demonstrationFeedbackPromise = queryDemonstrationFeedbackByInferenceId({
    inference_id,
    page_size: 1, // Only need to know if *any* exist
  });
  const feedbackBoundsPromise = queryFeedbackBoundsByTargetId({
    target_id: inference_id,
  });

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but time out and log a warning.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(inference_id, newFeedbackId, pageSize)
    : queryFeedbackByTargetId({
        target_id: inference_id,
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        page_size: pageSize,
      });

  // --- Execute all promises concurrently ---

  const [
    inference,
    model_inferences,
    dataset_counts,
    demonstration_feedback,
    feedback_bounds,
    feedback,
  ] = await Promise.all([
    inferencePromise,
    modelInferencesPromise,
    datasetCountsPromise,
    demonstrationFeedbackPromise,
    feedbackBoundsPromise,
    feedbackDataPromise,
  ]);

  // --- Process results ---

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
    newFeedbackId,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const _action = formData.get("_action");
  switch (_action) {
    case "addToDataset":
      return addToDataset(formData);
    case "addFeedback": {
      const response = await addHumanFeedback(formData);
      const url = new URL(request.url);
      url.searchParams.delete("beforeFeedback");
      url.searchParams.delete("afterFeedback");
      url.searchParams.set("newFeedbackId", response.feedback_id);
      return redirect(url.toString());
    }
    default:
      console.error(`Unknown action: ${_action}`);
      return null;
  }
}

async function addToDataset(formData: FormData) {
  const dataset = formData.get("dataset");
  const output = formData.get("output");
  const inference_id = formData.get("inference_id");
  if (!dataset || !output || !inference_id) {
    throw data("Missing required fields", { status: 400 });
  }
  try {
    const datapoint = await tensorZeroClient.createDatapoint(
      dataset.toString(),
      inference_id.toString(),
      output.toString() as "inherit" | "demonstration" | "none",
    );
    return redirect(
      `/datasets/${dataset.toString()}/datapoint/${datapoint.id}`,
    );
  } catch (error) {
    console.error(error);
    return data(
      "Failed to create datapoint as a datapoint exists with the same `source_inference_id`",
      { status: 400 },
    );
  }
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    hasDemonstration,
    newFeedbackId,
  } = loaderData;
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
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

  const config = useConfig();
  const variants = Object.keys(
    config.functions[inference.function_name]?.variants || {},
  );
  const addToDatasetFetcher = useFetcher();
  const actionError =
    addToDatasetFetcher.state === "idle" && addToDatasetFetcher.data
      ? addToDatasetFetcher.data
      : null;

  const handleAddToDataset = (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
  ) => {
    const formData = new FormData();
    formData.append("dataset", dataset);
    formData.append("output", output);
    formData.append("inference_id", inference.id);
    formData.append("_action", "addToDataset");
    addToDatasetFetcher.submit(formData, { method: "post", action: "." });
  };
  const { toast } = useToast();

  useEffect(() => {
    if (newFeedbackId) {
      toast({ title: "Feedback Added" });
    }
  }, [newFeedbackId, toast]);

  const variantInferenceFetcher = useInferenceActionFetcher();
  const variantSource = "inference";
  const variantInferenceIsLoading =
    // only concerned with rendering loading state when the modal is open
    isModalOpen &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
    const request = prepareInferenceActionRequest({
      resource: inference,
      source: variantSource,
      variant,
    });
    // TODO: handle JSON.stringify error
    submit({ data: JSON.stringify(request) });
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
  };

  return (
    <PageLayout>
      <PageHeader label="Inference" name={inference.id}>
        <BasicInfo
          inference={inference}
          inferenceUsage={getTotalInferenceUsage(model_inferences)}
        />

        {actionError && (
          <div className="mt-2 inline-block rounded-md bg-red-50 p-2 text-sm text-red-500">
            {typeof actionError === "string"
              ? actionError
              : "An unknown error occurred."}
          </div>
        )}

        <InferenceActions
          variants={variants}
          onVariantSelect={onVariantSelect}
          variantInferenceIsLoading={variantInferenceIsLoading}
          dataset_counts={dataset_counts}
          onDatasetSelect={handleAddToDataset}
          hasDemonstration={hasDemonstration}
          inferenceOutput={inference.output}
          inferenceId={inference.id}
        />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Input input={inference.input} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <Output
            output={inference.output}
            outputSchema={
              inference.function_type === "json"
                ? inference.output_schema
                : undefined
            }
          />
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

        {Object.keys(inference.tags).length > 0 && (
          <SectionLayout>
            <SectionHeader heading="Tags" />
            <TagsTable tags={inference.tags} />
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Model Inferences" />
          <ModelInferencesTable modelInferences={model_inferences} />
        </SectionLayout>
      </SectionsGroup>

      {selectedVariant && (
        <VariantResponseModal
          isOpen={isModalOpen}
          isLoading={variantInferenceIsLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={handleModalClose}
          item={inference}
          inferenceUsage={getTotalInferenceUsage(model_inferences)}
          selectedVariant={selectedVariant}
          source={variantSource}
        />
      )}
      <Toaster />
    </PageLayout>
  );
}

function getUserFacingError(error: unknown): {
  heading: string;
  message: ReactNode;
} {
  if (isRouteErrorResponse(error)) {
    switch (error.status) {
      case 400:
        return {
          heading: `${error.status}: Bad Request`,
          message: "Please try again later.",
        };
      case 401:
        return {
          heading: `${error.status}: Unauthorized`,
          message: "You do not have permission to access this resource.",
        };
      case 403:
        return {
          heading: `${error.status}: Forbidden`,
          message: "You do not have permission to access this resource.",
        };
      case 404:
        return {
          heading: `${error.status}: Not Found`,
          message:
            "The requested resource was not found. Please check the URL and try again.",
        };
      case 500:
      default:
        return {
          heading: "An unknown error occurred",
          message: "Please try again later.",
        };
    }
  }
  return {
    heading: "An unknown error occurred",
    message: "Please try again later.",
  };
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  const { heading, message } = getUserFacingError(error);
  return (
    <div className="flex flex-col items-center justify-center md:h-full">
      <div className="mt-8 flex flex-col items-center justify-center gap-2 rounded-xl border border-red-500 bg-red-50 p-6 md:mt-0">
        <h1 className="text-2xl font-bold">{heading}</h1>
        {typeof message === "string" ? <p>{message}</p> : message}
        <Link
          to={`/observability/inferences`}
          className="font-bold text-red-900 hover:text-red-800 hover:underline"
        >
          Go back
        </Link>
      </div>
    </div>
  );
}
