import type {
  ParsedInferenceRow,
  ParsedModelInferenceRow,
} from "~/utils/clickhouse/inference";
import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useConfig, useFunctionConfig } from "~/context/config";
import BasicInfo from "~/routes/observability/inferences/$inference_id/InferenceBasicInfo";
import Input from "~/components/inference/Input";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "~/routes/observability/inferences/$inference_id/InferenceParameters";
import { TagsTable } from "~/components/tags/TagsTable";
import { ModelInferencesTable } from "~/routes/observability/inferences/$inference_id/ModelInferencesTable";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
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
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";

export interface InferenceDetailData {
  inference: ParsedInferenceRow;
  model_inferences: ParsedModelInferenceRow[];
  feedback: FeedbackRow[];
  feedback_bounds: FeedbackBounds;
  hasDemonstration: boolean;
  latestFeedbackByMetric: Record<string, string>;
  usedVariants: string[];
}

/**
 * Props for rendering the header section (BasicInfo + ActionBar).
 * This allows the parent component to customize how the header is rendered,
 * e.g., wrapping it in a PageHeader for the full page view.
 */
export interface InferenceHeaderProps {
  basicInfo: ReactNode;
  actionBar: ReactNode;
}

interface InferenceDetailContentProps {
  data: InferenceDetailData;
  onFeedbackAdded?: (redirectUrl?: string) => void;
  feedbackFooter?: ReactNode;
  /**
   * Optional render function for customizing the header layout.
   * If not provided, BasicInfo and ActionBar are rendered directly.
   * Use this to wrap the header in a PageHeader for the full page view.
   */
  renderHeader?: (props: InferenceHeaderProps) => ReactNode;
}

type ModalType = "human-feedback" | "variant-response" | null;

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

export function InferenceDetailContent({
  data,
  onFeedbackAdded,
  feedbackFooter,
  renderHeader,
}: InferenceDetailContentProps) {
  const {
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
    hasDemonstration,
    latestFeedbackByMetric,
    usedVariants,
  } = data;

  const [openModal, setOpenModal] = useState<ModalType | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});

  const { toast } = useToast();

  // Demonstration feedback fetcher
  const demonstrationFeedbackFetcher = useFetcherWithReset<ActionData>();
  const {
    data: demonstrationFeedbackData,
    state: demonstrationFeedbackState,
    reset: resetDemonstrationFeedbackFetcher,
  } = demonstrationFeedbackFetcher;
  const demonstrationFeedbackFormError =
    demonstrationFeedbackState === "idle"
      ? (demonstrationFeedbackData?.error ?? null)
      : null;
  useEffect(() => {
    const currentState = demonstrationFeedbackState;
    const fetcherData = demonstrationFeedbackData;
    if (currentState === "idle" && fetcherData?.redirectTo) {
      onFeedbackAdded?.(fetcherData.redirectTo);
      setOpenModal(null);
      setSelectedVariant(null);
      resetDemonstrationFeedbackFetcher();
    }
  }, [
    demonstrationFeedbackData,
    demonstrationFeedbackState,
    resetDemonstrationFeedbackFetcher,
    onFeedbackAdded,
  ]);

  // Variant inference fetcher
  const variantInferenceFetcher = useInferenceActionFetcher();
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);
  const variantSource = "inference";
  const variantInferenceIsLoading =
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
        setSelectedVariant(null);
        setOpenModal(null);
      }
    } catch (error) {
      logger.error("Failed to prepare inference request:", error);

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

  // Human feedback fetcher
  const humanFeedbackFetcher = useFetcherWithReset<ActionData>();
  const {
    data: humanFeedbackData,
    state: humanFeedbackState,
    reset: resetHumanFeedbackFetcher,
  } = humanFeedbackFetcher;
  const humanFeedbackFormError =
    humanFeedbackState === "idle" ? (humanFeedbackData?.error ?? null) : null;
  useEffect(() => {
    const currentState = humanFeedbackState;
    const fetcherData = humanFeedbackData;
    if (currentState === "idle" && fetcherData?.redirectTo) {
      onFeedbackAdded?.(fetcherData.redirectTo);
      setOpenModal(null);
      resetHumanFeedbackFetcher();
    }
  }, [
    humanFeedbackData,
    humanFeedbackState,
    onFeedbackAdded,
    resetHumanFeedbackFetcher,
  ]);

  const config = useConfig();

  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([
    ...usedVariants,
    ...Object.keys(config.models),
  ]);
  const models = [...modelsSet].sort();

  const options = isDefault ? models : variants;
  const onSelect = isDefault ? onModelSelect : onVariantSelect;

  const inferenceUsage = getTotalInferenceUsage(model_inferences);

  // Build the header components
  const basicInfoElement = (
    <BasicInfo
      inference={inference}
      inferenceUsage={inferenceUsage}
      modelInferences={model_inferences}
    />
  );

  const actionBarElement = (
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
          if (humanFeedbackState !== "idle") {
            return;
          }

          if (!isOpen) {
            resetHumanFeedbackFetcher();
          }
          setOpenModal(isOpen ? "human-feedback" : null);
        }}
        isOpen={openModal === "human-feedback"}
        trigger={<HumanFeedbackButton />}
      >
        <humanFeedbackFetcher.Form method="post" action="/api/feedback">
          <HumanFeedbackForm
            inferenceId={inference.id}
            inferenceOutput={inference.output}
            formError={humanFeedbackFormError}
            isSubmitting={
              humanFeedbackState === "submitting" ||
              humanFeedbackState === "loading"
            }
          />
        </humanFeedbackFetcher.Form>
      </HumanFeedbackModal>
    </ActionBar>
  );

  // Allow parent to customize header rendering, or use default layout
  const headerContent = renderHeader ? (
    renderHeader({ basicInfo: basicInfoElement, actionBar: actionBarElement })
  ) : (
    <>
      {basicInfoElement}
      {actionBarElement}
    </>
  );

  return (
    <>
      {headerContent}

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
            count={feedback.length}
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
          {feedbackFooter}
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
          inferenceUsage={inferenceUsage}
          selectedVariant={selectedVariant}
          source={variantSource}
          onRefresh={lastRequestArgs ? handleRefresh : null}
        >
          {variantInferenceFetcher.data?.info && (
            <demonstrationFeedbackFetcher.Form
              method="post"
              action="/api/feedback"
            >
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
                isSubmitting={demonstrationFeedbackState === "submitting"}
                submissionError={demonstrationFeedbackFormError}
              />
            </demonstrationFeedbackFetcher.Form>
          )}
        </VariantResponseModal>
      )}
    </>
  );
}

function prepareDemonstrationFromVariantOutput(
  variantOutput: VariantResponseInfo,
) {
  const output = variantOutput.output;
  if (Array.isArray(output)) {
    return output;
  } else if (output && "parsed" in output) {
    return output.parsed;
  } else {
    throw new Error("Invalid variant output");
  }
}
