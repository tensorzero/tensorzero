import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  resolveModelInferences,
  loadFileDataForStoredInput,
} from "~/utils/resolve.server";
import type { Route } from "./+types/route";
import {
  Await,
  data,
  useAsyncError,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { Suspense, useCallback, useEffect, useState } from "react";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type {
  FeedbackRow,
  FeedbackBounds,
  StoredInference,
  Input,
} from "~/types/tensorzero";
import {
  isJsonOutput,
  type ParsedModelInferenceRow,
} from "~/utils/clickhouse/inference";

// Section components
import BasicInfo from "./InferenceBasicInfo";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "./InferenceParameters";
import { ToolParametersSection } from "~/components/inference/ToolParametersSection";
import { TagsTable } from "~/components/tags/TagsTable";
import { ModelInferencesTable } from "./ModelInferencesTable";
import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithButton } from "~/components/inference/TryWithButton";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { DemonstrationFeedbackButton } from "~/components/feedback/DemonstrationFeedbackButton";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
  type VariantResponseInfo,
} from "~/routes/api/tensorzero/inference.utils";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { useConfig, useFunctionConfig } from "~/context/config";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

// Types for deferred data
export type ModelInferencesData = ParsedModelInferenceRow[];

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedback_bounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
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

  const tensorZeroClient = getTensorZeroClient();

  // --- PRIMARY DATA: Await these (needed for page to render) ---
  const inferencesResponse = await tensorZeroClient.getInferences({
    ids: [inference_id],
    output_source: "inference",
  });

  if (inferencesResponse.inferences.length !== 1) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }
  const inference = inferencesResponse.inferences[0];

  // Resolve input (also primary - needed for Input section)
  const resolvedInput = await loadFileDataForStoredInput(inference.input);

  // Get hasDemonstration and usedVariants (needed for action bar)
  const [demonstrationFeedback, usedVariants] = await Promise.all([
    tensorZeroClient.getDemonstrationFeedback(inference_id, { limit: 1 }),
    inference.function_name === DEFAULT_FUNCTION
      ? tensorZeroClient.getUsedVariants(inference.function_name)
      : Promise.resolve([]),
  ]);
  const hasDemonstration = demonstrationFeedback.length > 0;

  // --- SECONDARY DATA: Return as promises (stream in later) ---

  // Model inferences promise
  const modelInferencesPromise: Promise<ModelInferencesData> = tensorZeroClient
    .getModelInferences(inference_id)
    .then((response) => resolveModelInferences(response.model_inferences));

  // Feedback data promise - handles the newFeedbackId polling case
  const feedbackDataPromise: Promise<FeedbackData> = newFeedbackId
    ? // Sequential case: poll first, then query bounds/metrics
      pollForFeedbackItem(inference_id, newFeedbackId, limit).then(
        async (feedback) => {
          const [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
            tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
            tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
          ]);
          return { feedback, feedback_bounds, latestFeedbackByMetric };
        },
      )
    : // Normal case: execute all queries in parallel
      Promise.all([
        tensorZeroClient.getFeedbackByTargetId(inference_id, {
          before: beforeFeedback || undefined,
          after: afterFeedback || undefined,
          limit,
        }),
        tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
        tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
      ]).then(([feedback, feedback_bounds, latestFeedbackByMetric]) => ({
        feedback,
        feedback_bounds,
        latestFeedbackByMetric,
      }));

  return {
    // Primary (resolved)
    inference,
    resolvedInput,
    hasDemonstration,
    usedVariants,
    newFeedbackId,
    // Secondary (deferred promises)
    modelInferencesPromise,
    feedbackDataPromise,
  };
}

// --- Skeleton Components ---

function ModelInferencesSkeleton() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Provider</TableHead>
          <TableHead>Input Tokens</TableHead>
          <TableHead>Output Tokens</TableHead>
          <TableHead>Response Time</TableHead>
          <TableHead>TTFT</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {Array.from({ length: 2 }).map((_, i) => (
          <TableRow key={i}>
            <TableCell>
              <Skeleton className="h-4 w-24" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-20" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function FeedbackSkeleton() {
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Metric</TableHead>
            <TableHead>Value</TableHead>
            <TableHead>Tags</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array.from({ length: 3 }).map((_, i) => (
            <TableRow key={i}>
              <TableCell>
                <Skeleton className="h-4 w-24" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-20" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-16" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-24" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-28" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <PageButtons
        onPreviousPage={() => {}}
        onNextPage={() => {}}
        disablePrevious
        disableNext
      />
    </>
  );
}

// --- Error Components ---

function ModelInferencesError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load model inferences";

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Provider</TableHead>
          <TableHead>Input Tokens</TableHead>
          <TableHead>Output Tokens</TableHead>
          <TableHead>Response Time</TableHead>
          <TableHead>TTFT</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <TableRow>
          <TableCell colSpan={6} className="text-center">
            <div className="flex flex-col items-center gap-2 py-8 text-red-600">
              <span className="font-medium">Error loading data</span>
              <span className="text-muted-foreground text-sm">{message}</span>
            </div>
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}

function FeedbackSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load feedback";

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Metric</TableHead>
            <TableHead>Value</TableHead>
            <TableHead>Tags</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell colSpan={5} className="text-center">
              <div className="flex flex-col items-center gap-2 py-8 text-red-600">
                <span className="font-medium">Error loading data</span>
                <span className="text-muted-foreground text-sm">{message}</span>
              </div>
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
      <PageButtons
        onPreviousPage={() => {}}
        onNextPage={() => {}}
        disablePrevious
        disableNext
      />
    </>
  );
}

// --- Content Components ---

function ModelInferencesContent({ data }: { data: ModelInferencesData }) {
  return <ModelInferencesTable modelInferences={data} />;
}

function FeedbackSectionContent({ data }: { data: FeedbackData }) {
  const { feedback, feedback_bounds, latestFeedbackByMetric } = data;
  const navigate = useNavigate();

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

  return (
    <>
      <FeedbackTable
        feedback={feedback}
        latestCommentId={feedback_bounds.by_type.comment.last_id!}
        latestDemonstrationId={feedback_bounds.by_type.demonstration.last_id!}
        latestFeedbackIdByMetric={latestFeedbackByMetric}
      />
      <PageButtons
        onNextPage={handleNextFeedbackPage}
        onPreviousPage={handlePreviousFeedbackPage}
        disableNext={disableNextFeedbackPage}
        disablePrevious={disablePreviousFeedbackPage}
      />
    </>
  );
}

// --- Action Bar Component ---

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

interface InferenceActionBarProps {
  inference: StoredInference;
  hasDemonstration: boolean;
  usedVariants: string[];
  onFeedbackAdded?: (redirectUrl?: string) => void;
  onVariantSelect: (variant: string) => void;
  isVariantLoading: boolean;
}

function InferenceActionBar({
  inference,
  hasDemonstration,
  usedVariants,
  onFeedbackAdded,
  onVariantSelect,
  isVariantLoading,
}: InferenceActionBarProps) {
  const [isHumanFeedbackModalOpen, setIsHumanFeedbackModalOpen] =
    useState(false);
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const config = useConfig();

  const isDefault = inference.function_name === DEFAULT_FUNCTION;
  const modelsSet = new Set<string>([
    ...usedVariants,
    ...(config?.model_names ?? []),
  ]);
  const models = [...modelsSet].sort();
  const options = isDefault ? models : variants;

  const humanFeedbackFetcher = useFetcherWithReset<ActionData>();
  const humanFeedbackFormError =
    humanFeedbackFetcher.state === "idle"
      ? (humanFeedbackFetcher.data?.error ?? null)
      : null;

  useEffect(() => {
    if (
      humanFeedbackFetcher.state === "idle" &&
      humanFeedbackFetcher.data?.redirectTo
    ) {
      onFeedbackAdded?.(humanFeedbackFetcher.data.redirectTo);
      setIsHumanFeedbackModalOpen(false);
      humanFeedbackFetcher.reset();
    }
  }, [humanFeedbackFetcher, onFeedbackAdded]);

  return (
    <ActionBar>
      <TryWithButton
        options={options}
        onSelect={onVariantSelect}
        isLoading={isVariantLoading}
        isDefaultFunction={isDefault}
      />
      <AddToDatasetButton
        inferenceId={inference.inference_id}
        functionName={inference.function_name}
        variantName={inference.variant_name}
        episodeId={inference.episode_id}
        hasDemonstration={hasDemonstration}
      />
      <HumanFeedbackModal
        onOpenChange={(isOpen) => {
          if (humanFeedbackFetcher.state !== "idle") return;
          if (!isOpen) humanFeedbackFetcher.reset();
          setIsHumanFeedbackModalOpen(isOpen);
        }}
        isOpen={isHumanFeedbackModalOpen}
        trigger={<HumanFeedbackButton />}
      >
        <humanFeedbackFetcher.Form method="post" action="/api/feedback">
          <HumanFeedbackForm
            inferenceId={inference.inference_id}
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
  );
}

// --- Helper Functions ---

function prepareDemonstrationFromVariantOutput(
  variantOutput: VariantResponseInfo,
) {
  const output = variantOutput.output;
  if (output === undefined) {
    return undefined;
  }
  if (isJsonOutput(output)) {
    return output.parsed;
  }
  return output;
}

// --- Main Page Component ---

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    resolvedInput,
    hasDemonstration,
    usedVariants,
    newFeedbackId,
    modelInferencesPromise,
    feedbackDataPromise,
  } = loaderData;

  const navigate = useNavigate();
  const { toast } = useToast();

  // Track resolved model inferences for usage calculation in modal
  const [resolvedModelInferences, setResolvedModelInferences] = useState<
    ParsedModelInferenceRow[] | null
  >(null);

  // Resolve model inferences promise when it completes
  useEffect(() => {
    modelInferencesPromise
      .then(setResolvedModelInferences)
      .catch(() => setResolvedModelInferences(null));
  }, [modelInferencesPromise]);

  // Variant response modal state
  const [isVariantModalOpen, setIsVariantModalOpen] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);

  // Variant inference fetcher
  const variantInferenceFetcher = useInferenceActionFetcher();
  const variantSource = "inference";
  const variantInferenceIsLoading =
    isVariantModalOpen &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const processVariantRequest = (
    option: string,
    args: Parameters<typeof prepareInferenceActionRequest>[0],
  ) => {
    try {
      const request = prepareInferenceActionRequest(args);

      setSelectedVariant(option);
      setIsVariantModalOpen(true);
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
        setIsVariantModalOpen(false);
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

  // Check if function is default to determine handler type
  const isDefaultFunction = inference.function_name === DEFAULT_FUNCTION;

  const handleVariantOrModelSelect = (option: string) => {
    processVariantRequest(option, {
      resource: inference,
      input: resolvedInput as Input,
      source: variantSource,
      ...(isDefaultFunction ? { model_name: option } : { variant: option }),
    });
  };

  const handleVariantRefresh = () => {
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

  // Handle feedback added callback
  const handleFeedbackAdded = useCallback(
    (redirectUrl?: string) => {
      if (redirectUrl) {
        const url = new URL(redirectUrl, window.location.origin);
        const newFeedbackIdParam = url.searchParams.get("newFeedbackId");
        if (newFeedbackIdParam) {
          const currentUrl = new URL(window.location.href);
          currentUrl.searchParams.delete("beforeFeedback");
          currentUrl.searchParams.delete("afterFeedback");
          currentUrl.searchParams.set("newFeedbackId", newFeedbackIdParam);
          navigate(currentUrl.pathname + currentUrl.search);
        }
      }
    },
    [navigate],
  );

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
      handleFeedbackAdded(fetcherData.redirectTo);
      setIsVariantModalOpen(false);
      setSelectedVariant(null);
      resetDemonstrationFeedbackFetcher();
    }
  }, [
    demonstrationFeedbackData,
    demonstrationFeedbackState,
    resetDemonstrationFeedbackFetcher,
    handleFeedbackAdded,
  ]);

  // Show toast when feedback is successfully added
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  return (
    <PageLayout>
      <PageHeader label="Inference" name={inference.inference_id}>
        <BasicInfo
          inference={inference}
          modelInferencesPromise={modelInferencesPromise}
        />
        <InferenceActionBar
          inference={inference}
          hasDemonstration={hasDemonstration}
          usedVariants={usedVariants}
          onFeedbackAdded={handleFeedbackAdded}
          onVariantSelect={handleVariantOrModelSelect}
          isVariantLoading={variantInferenceIsLoading}
        />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputElement input={resolvedInput} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          {inference.type === "json" ? (
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
            badge={{
              name: "inference",
              tooltip:
                "This table only includes inference-level feedback. To see episode-level feedback, open the detail page for that episode.",
            }}
          />
          <Suspense fallback={<FeedbackSkeleton />}>
            <Await
              resolve={feedbackDataPromise}
              errorElement={<FeedbackSectionError />}
            >
              {(feedbackData) => <FeedbackSectionContent data={feedbackData} />}
            </Await>
          </Suspense>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Inference Parameters" />
          <ParameterCard
            parameters={JSON.stringify(inference.inference_params, null, 2)}
          />
        </SectionLayout>

        {inference.type === "chat" && (
          <SectionLayout>
            <SectionHeader heading="Tool Parameters" />
            <ToolParametersSection
              allowed_tools={inference.allowed_tools}
              additional_tools={inference.additional_tools}
              tool_choice={inference.tool_choice}
              parallel_tool_calls={inference.parallel_tool_calls}
              provider_tools={inference.provider_tools}
            />
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsTable
            tags={Object.fromEntries(
              Object.entries(inference.tags).filter(
                (entry): entry is [string, string] => entry[1] !== undefined,
              ),
            )}
            isEditing={false}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Model Inferences" />
          <Suspense fallback={<ModelInferencesSkeleton />}>
            <Await
              resolve={modelInferencesPromise}
              errorElement={<ModelInferencesError />}
            >
              {(modelInferences) => (
                <ModelInferencesContent data={modelInferences} />
              )}
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>

      {selectedVariant && (
        <VariantResponseModal
          isOpen={isVariantModalOpen}
          isLoading={variantInferenceIsLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={() => {
            setIsVariantModalOpen(false);
            setSelectedVariant(null);
            setLastRequestArgs(null);
          }}
          item={inference}
          inferenceUsage={
            resolvedModelInferences
              ? getTotalInferenceUsage(resolvedModelInferences)
              : undefined
          }
          selectedVariant={selectedVariant}
          source={variantSource}
          onRefresh={lastRequestArgs ? handleVariantRefresh : null}
        >
          {variantInferenceFetcher.data?.info && (
            <demonstrationFeedbackFetcher.Form
              method="post"
              action="/api/feedback"
            >
              <input type="hidden" name="metricName" value="demonstration" />
              <input
                type="hidden"
                name="inferenceId"
                value={inference.inference_id}
              />
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
    </PageLayout>
  );
}
