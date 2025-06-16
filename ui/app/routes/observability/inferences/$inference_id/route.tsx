import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference.server";
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
  useFetcher,
  useNavigate,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import BasicInfo from "./InferenceBasicInfo";
import InputSnippet from "~/components/inference/InputSnippet";
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
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";
import { Toaster } from "~/components/ui/toaster";
import { useToast } from "~/hooks/use-toast";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";
import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithVariantButton } from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { isServerRequestError } from "~/utils/common";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { codeToHtml } from "shiki";

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

  const [inferenceParams, toolParams] = await Promise.all([
    processJson(
      inference.inference_params,
      `inference params: ${inference_id}`,
    ),
    inference.function_type === "chat"
      ? processJson(inference.tool_params, `tool params: ${inference_id}`)
      : Promise.resolve(null),
  ]);

  return {
    inference,
    inferenceParams,
    toolParams,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    hasDemonstration: demonstration_feedback.length > 0,
    newFeedbackId,
  };
}

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const _action = formData.get("_action");
  switch (_action) {
    case "addToDataset": {
      const dataset = formData.get("dataset");
      const output = formData.get("output");
      const inference_id = formData.get("inference_id");
      if (!dataset || !output || !inference_id) {
        return data<ActionData>(
          { error: "Missing required fields" },
          { status: 400 },
        );
      }
      try {
        const datapoint = await tensorZeroClient.createDatapoint(
          dataset.toString(),
          inference_id.toString(),
          output.toString() as "inherit" | "demonstration" | "none",
        );
        return data<ActionData>({
          redirectTo: `/datasets/${dataset.toString()}/datapoint/${datapoint.id}`,
        });
      } catch (error) {
        console.error(error);
        return data<ActionData>(
          {
            error:
              "Failed to create datapoint as a datapoint exists with the same `source_inference_id`",
          },
          { status: 400 },
        );
      }
    }
    case "addFeedback": {
      try {
        const response = await addHumanFeedback(formData);
        const url = new URL(request.url);
        url.searchParams.delete("beforeFeedback");
        url.searchParams.delete("afterFeedback");
        url.searchParams.set("newFeedbackId", response.feedback_id);
        return data<ActionData>({ redirectTo: url.pathname + url.search });
      } catch (error) {
        if (isServerRequestError(error)) {
          return data<ActionData>(
            { error: error.message },
            { status: error.statusCode },
          );
        }
        return data<ActionData>(
          { error: "Unknown server error. Try again." },
          { status: 500 },
        );
      }
    }
    default:
      console.error(`Unknown action: ${_action}`);
      return data<ActionData>(
        { error: "Unknown server action" },
        { status: 400 },
      );
  }
}

type ModalType = "human-feedback" | "variant-response" | null;

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    inferenceParams,
    toolParams,
    model_inferences,
    feedback,
    feedback_bounds,
    dataset_counts,
    hasDemonstration,
    newFeedbackId,
  } = loaderData;
  const navigate = useNavigate();
  const [openModal, setOpenModal] = useState<ModalType | null>(null);
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
  const addToDatasetFetcher = useFetcher<typeof action>();
  const addToDatasetError =
    addToDatasetFetcher.state === "idle" && addToDatasetFetcher.data?.error
      ? addToDatasetFetcher.data.error
      : null;
  useEffect(() => {
    const currentState = addToDatasetFetcher.state;
    const data = addToDatasetFetcher.data;
    if (currentState === "idle" && data?.redirectTo) {
      navigate(data.redirectTo);
    }
  }, [addToDatasetFetcher.data, addToDatasetFetcher.state, navigate]);

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
    openModal === "variant-response" &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setOpenModal("variant-response");
    const request = prepareInferenceActionRequest({
      resource: inference,
      source: variantSource,
      variant,
    });
    // TODO: handle JSON.stringify error
    submit({ data: JSON.stringify(request) });
  };

  const humanFeedbackFetcher = useFetcherWithReset<typeof action>();
  const humanFeedbackFormError =
    humanFeedbackFetcher.state === "idle"
      ? (humanFeedbackFetcher.data?.error ?? null)
      : null;
  useEffect(() => {
    const currentState = humanFeedbackFetcher.state;
    const data = humanFeedbackFetcher.data;
    if (currentState === "idle" && data?.redirectTo) {
      navigate(data.redirectTo, { state: "humanFeedbackRedirect" });
      setOpenModal(null);
    }
  }, [humanFeedbackFetcher.data, humanFeedbackFetcher.state, navigate]);

  return (
    <PageLayout>
      <PageHeader label="Inference" name={inference.id}>
        <BasicInfo
          inference={inference}
          inferenceUsage={getTotalInferenceUsage(model_inferences)}
          modelInferences={model_inferences}
        />

        {addToDatasetError && (
          <div className="mt-2 inline-block rounded-md bg-red-50 p-2 text-sm text-red-500">
            {addToDatasetError}
          </div>
        )}

        <ActionBar>
          <TryWithVariantButton
            variants={variants}
            onVariantSelect={onVariantSelect}
            isLoading={variantInferenceIsLoading}
          />
          <AddToDatasetButton
            dataset_counts={dataset_counts}
            onDatasetSelect={handleAddToDataset}
            hasDemonstration={hasDemonstration}
          />
          <HumanFeedbackModal
            onOpenChange={(isOpen) => {
              if (humanFeedbackFetcher.state !== "idle") {
                return;
              }

              if (!isOpen) {
                humanFeedbackFetcher.reset();
              }
              setOpenModal(isOpen ? "human-feedback" : null);
            }}
            isOpen={openModal === "human-feedback"}
            trigger={<HumanFeedbackButton />}
          >
            <humanFeedbackFetcher.Form method="post">
              <input type="hidden" name="_action" value="addFeedback" />
              <HumanFeedbackForm
                inferenceId={inference.id}
                inferenceOutput={inference.output}
                formError={humanFeedbackFormError}
                isSubmitting={
                  humanFeedbackFetcher.state === "submitting" ||
                  humanFeedbackFetcher.state === "loading"
                }
              />
            </humanFeedbackFetcher.Form>
          </HumanFeedbackModal>
        </ActionBar>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputSnippet input={inference.input} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <Output
            output={
              inference.function_type === "json"
                ? { ...inference.output, schema: inference.output_schema }
                : inference.output
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
          <ParameterCard parameters={inferenceParams.raw} />
        </SectionLayout>

        {inference.function_type === "chat" && (
          <SectionLayout>
            <SectionHeader heading="Tool Parameters" />
            {toolParams && <ParameterCard parameters={toolParams.raw} />}
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsTable tags={inference.tags} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Model Inferences" />
          <ModelInferencesTable modelInferences={model_inferences} />
        </SectionLayout>
      </SectionsGroup>

      {selectedVariant && (
        <VariantResponseModal
          isOpen={openModal === "variant-response"}
          isLoading={variantInferenceIsLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={() => {
            setOpenModal(null);
            setSelectedVariant(null);
          }}
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
      <div className="mt-8 flex flex-col items-center justify-center gap-2 rounded-xl bg-red-50 p-6 md:mt-0">
        <h1 className="text-2xl font-bold">{heading}</h1>
        {typeof message === "string" ? <p>{message}</p> : message}
        <Link
          to={`/observability/inferences`}
          className="font-bold text-red-800 hover:text-red-600"
        >
          Go back &rarr;
        </Link>
      </div>
    </div>
  );
}

async function processJson(object: object, objectRef: string) {
  let raw: string | null = null;
  let html: string | null = null;
  try {
    raw = JSON.stringify(object, null, 2);
    html = await highlightCode(raw).catch((error) => {
      console.error(
        `Syntax error highlighting error for ${objectRef}. Using raw JSON instead.`,
        error,
      );
      return null;
    });
  } catch {
    throw data(`Failed to parse JSON for ${objectRef}`, { status: 500 });
  }

  return { raw, html };
}

async function highlightCode(code: string, args?: { lang?: "json" }) {
  const { lang = "json" } = args || {};
  return await codeToHtml(code, { lang, theme: "vitesse-dark" });
}
