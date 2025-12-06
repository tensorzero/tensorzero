import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference.server";
import {
  pollForFeedbackItem,
  queryLatestFeedbackIdByMetric,
} from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  Link,
  useFetcher,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import BasicInfo from "./InferenceBasicInfo";
import Input from "~/components/inference/Input";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { ParameterCard } from "./InferenceParameters";
import { TagsTable } from "~/components/tags/TagsTable";
import { ModelInferencesTable } from "./ModelInferencesTable";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useConfig, useFunctionConfig } from "~/context/config";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
  type VariantResponseInfo,
} from "~/routes/api/tensorzero/inference.utils";
import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithButton } from "~/components/inference/TryWithButton";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { DemonstrationFeedbackButton } from "~/components/feedback/DemonstrationFeedbackButton";
import { logger } from "~/utils/logger";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { getUsedVariants } from "~/utils/clickhouse/function";
import { DEFAULT_FUNCTION } from "~/utils/constants";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const limit = Number(url.searchParams.get("limit")) || 10;

  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  // --- Define all promises, conditionally choosing the feedback promise ---

  const dbClient = await getNativeDatabaseClient();

  const inferencePromise = queryInferenceById(inference_id);
  const modelInferencesPromise =
    queryModelInferencesByInferenceId(inference_id);
  const demonstrationFeedbackPromise =
    dbClient.queryDemonstrationFeedbackByInferenceId({
      inference_id,
      limit: 1, // Only need to know if *any* exist
    });
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but eventually time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(inference_id, newFeedbackId, limit)
    : dbClient.queryFeedbackByTargetId({
        target_id: inference_id,
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      });

  // --- Execute promises concurrently (with special handling for new feedback) ---

  let inference,
    model_inferences,
    demonstration_feedback,
    feedback_bounds,
    feedback,
    latestFeedbackByMetric;

  if (newFeedbackId) {
    // When there's new feedback, wait for polling to complete before querying
    // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
    [inference, model_inferences, demonstration_feedback, feedback] =
      await Promise.all([
        inferencePromise,
        modelInferencesPromise,
        demonstrationFeedbackPromise,
        feedbackDataPromise,
      ]);

    // Query these after polling completes to avoid race condition with materialized views
    [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
      dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
      queryLatestFeedbackIdByMetric({ target_id: inference_id }),
    ]);
  } else {
    // Normal case: execute all queries in parallel
    [
      inference,
      model_inferences,
      demonstration_feedback,
      feedback_bounds,
      feedback,
      latestFeedbackByMetric,
    ] = await Promise.all([
      inferencePromise,
      modelInferencesPromise,
      demonstrationFeedbackPromise,
      dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
      feedbackDataPromise,
      queryLatestFeedbackIdByMetric({ target_id: inference_id }),
    ]);
  }

  // --- Process results ---

  if (!inference) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }

  const usedVariants =
    inference.function_name === DEFAULT_FUNCTION
      ? await getUsedVariants(inference.function_name)
      : [];

  return {
    inference,
    model_inferences,
    usedVariants,
    feedback,
    feedback_bounds,
    hasDemonstration: demonstration_feedback.length > 0,
    newFeedbackId,
    latestFeedbackByMetric,
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
      return handleAddToDatasetAction(formData);
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
        if (isTensorZeroServerError(error)) {
          return data<ActionData>(
            { error: error.message },
            { status: error.status },
          );
        }
        return data<ActionData>(
          { error: "Unknown server error. Try again." },
          { status: 500 },
        );
      }
    }
    case null:
      logger.error("No action provided");
      return data<ActionData>({ error: "No action provided" }, { status: 400 });
    default:
      logger.error(`Unknown action: ${_action}`);
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
    model_inferences,
    usedVariants,
    feedback,
    feedback_bounds,
    hasDemonstration,
    newFeedbackId,
    latestFeedbackByMetric,
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

  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});

  const demonstrationFeedbackFetcher = useFetcher<typeof action>();
  const demonstrationFeedbackFormError =
    demonstrationFeedbackFetcher.state === "idle"
      ? (demonstrationFeedbackFetcher.data?.error ?? null)
      : null;
  useEffect(() => {
    const currentState = demonstrationFeedbackFetcher.state;
    const data = demonstrationFeedbackFetcher.data;
    if (currentState === "idle" && data?.redirectTo) {
      navigate(data.redirectTo);
      setOpenModal(null);
      setSelectedVariant(null);
    }
  }, [
    demonstrationFeedbackFetcher.data,
    demonstrationFeedbackFetcher.state,
    navigate,
  ]);

  const { toast } = useToast();

  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  const variantInferenceFetcher = useInferenceActionFetcher();
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);
  const variantSource = "inference";
  const variantInferenceIsLoading =
    // only concerned with rendering loading state when the modal is open
    openModal === "variant-response" &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const processRequest = (
    option: string,
    args: Parameters<typeof prepareInferenceActionRequest>[0],
  ) => {
    try {
      const request = prepareInferenceActionRequest(args);

      // Set state and open modal only if request preparation succeeds
      setSelectedVariant(option);
      setOpenModal("variant-response");
      setLastRequestArgs(args);

      try {
        void submit({ data: JSON.stringify(request) });
      } catch (stringifyError) {
        logger.error("Failed to stringify request:", stringifyError);
        toast.error({
          title: "Request Error",
          description: "Failed to prepare the request. Please try again.",
        });
        // Reset state on error
        setSelectedVariant(null);
        setOpenModal(null);
      }
    } catch (error) {
      logger.error("Failed to prepare inference request:", error);

      // Show user-friendly error message based on the error type
      let errorMessage = "Failed to prepare the request. Please try again.";
      if (error instanceof Error) {
        if (error.message.includes("Extra body is not supported")) {
          errorMessage =
            "This inference contains extra body parameters which are not supported in the UI.";
        } else if (error.message) {
          errorMessage = error.message;
        }
      }

      toast.error({
        title: "Request Preparation Error",
        description: errorMessage,
      });
    }
  };

  const onVariantSelect = (variant: string) => {
    processRequest(variant, {
      resource: inference,
      source: variantSource,
      variant,
    });
  };

  const onModelSelect = (model: string) => {
    processRequest(model, {
      resource: inference,
      source: variantSource,
      model_name: model,
    });
  };

  const handleRefresh = () => {
    if (!lastRequestArgs) {
      return;
    }

    try {
      const request = prepareInferenceActionRequest(lastRequestArgs);
      request.cache_options = {
        ...request.cache_options,
        enabled: "write_only",
      };
      submit({ data: JSON.stringify(request) });
    } catch (error) {
      logger.error("Failed to prepare inference request for refresh:", error);
      toast.error({
        title: "Request Preparation Error",
        description: "Failed to refresh inference. Please try again.",
      });
    }
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

  const config = useConfig();

  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([
    // models successfully used with default function
    ...usedVariants,
    // all configured models in config
    ...config.model_names,
    // TODO(bret): list of popular/common model choices
    // see https://github.com/tensorzero/tensorzero/issues/1396#issuecomment-3286424944
  ]);
  const models = [...modelsSet].sort();

  const options = isDefault ? models : variants;
  const onSelect = isDefault ? onModelSelect : onVariantSelect;

  return (
    <PageLayout>
      <PageHeader label="Inference" name={inference.id}>
        <BasicInfo
          inference={inference}
          inferenceUsage={getTotalInferenceUsage(model_inferences)}
          modelInferences={model_inferences}
        />

        <ActionBar>
          <TryWithButton
            options={options}
            onOptionSelect={onSelect}
            isLoading={variantInferenceIsLoading}
            isDefaultFunction={isDefault}
          />
          <AddToDatasetButton
            inferenceId={inference.id}
            functionName={inference.function_name}
            variantName={inference.variant_name}
            episodeId={inference.episode_id}
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
          <Input
            system={inference.input.system}
            messages={inference.input.messages}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          {inference.function_type === "json" ? (
            <JsonOutputElement
              output={inference.output}
              outputSchema={inference.output_schema}
            />
          ) : (
            <ChatOutputElement output={inference.output} />
          )}
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
          <FeedbackTable
            feedback={feedback}
            latestCommentId={feedback_bounds.by_type.comment.last_id!}
            latestDemonstrationId={
              feedback_bounds.by_type.demonstration.last_id!
            }
            latestFeedbackIdByMetric={latestFeedbackByMetric}
          />
          <PageButtons
            onNextPage={handleNextFeedbackPage}
            onPreviousPage={handlePreviousFeedbackPage}
            disableNext={disableNextFeedbackPage}
            disablePrevious={disablePreviousFeedbackPage}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Inference Parameters" />
          <ParameterCard
            parameters={JSON.stringify(inference.inference_params, null, 2)}
          />
        </SectionLayout>

        {inference.function_type === "chat" && (
          <SectionLayout>
            <SectionHeader heading="Tool Parameters" />
            {inference.tool_params && (
              <ParameterCard
                parameters={JSON.stringify(inference.tool_params, null, 2)}
              />
            )}
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsTable tags={inference.tags} isEditing={false} />
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
            setLastRequestArgs(null);
          }}
          item={inference}
          inferenceUsage={getTotalInferenceUsage(model_inferences)}
          selectedVariant={selectedVariant}
          source={variantSource}
          onRefresh={lastRequestArgs ? handleRefresh : null}
        >
          {variantInferenceFetcher.data?.info && (
            <demonstrationFeedbackFetcher.Form method="post">
              <input type="hidden" name="_action" value="addFeedback" />
              <input type="hidden" name="metricName" value="demonstration" />
              <input type="hidden" name="inferenceId" value={inference.id} />
              <input
                type="hidden"
                name="value"
                value={JSON.stringify(
                  prepareDemonstrationFromVariantOutput(
                    variantInferenceFetcher.data.info,
                  ),
                )}
              />
              <DemonstrationFeedbackButton
                isSubmitting={
                  demonstrationFeedbackFetcher.state === "submitting"
                }
                submissionError={demonstrationFeedbackFormError}
              />
            </demonstrationFeedbackFetcher.Form>
          )}
        </VariantResponseModal>
      )}
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
    logger.error(error);
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

function prepareDemonstrationFromVariantOutput(
  variantOutput: VariantResponseInfo,
) {
  const output = variantOutput.output;
  // output can either be a JsonInferenceOutput or a ContentBlockChatOutput[] (or undefined)
  // if it is a JsonInferenceOutput, we need to take the Parsed field and throw if it is missing
  // if it is a ContentBlockChatOutput[], we can return as is
  if (Array.isArray(output)) {
    return output;
  } else if (output && "parsed" in output) {
    return output.parsed;
  } else {
    throw new Error("Invalid variant output");
  }
}
