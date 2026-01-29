import React, { Suspense, useEffect, useState, useCallback } from "react";
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
  useLocation,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import type {
  FeedbackRow,
  FeedbackBounds,
  StoredInference,
  Input,
} from "~/types/tensorzero";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import { Skeleton } from "~/components/ui/skeleton";
import BasicInfo from "./InferenceBasicInfo";
import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithSelect } from "~/components/inference/TryWithSelect";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { useConfig, useFunctionConfig } from "~/context/config";
import { DemonstrationFeedbackButton } from "~/components/feedback/DemonstrationFeedbackButton";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
  type VariantResponseInfo,
} from "~/routes/api/tensorzero/inference.utils";
import { logger } from "~/utils/logger";
import { isJsonOutput } from "~/utils/clickhouse/inference";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "./InferenceParameters";
import { ToolParametersSection } from "~/components/inference/ToolParametersSection";
import { TagsTable } from "~/components/tags/TagsTable";
import { ModelInferencesTable } from "./ModelInferencesTable";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { TableErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

// Types for streamed data
export type ModelInferencesData = ParsedModelInferenceRow[];

export type ActionBarData = {
  hasDemonstration: boolean;
  usedVariants: string[];
};

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedback_bounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
};

// Fetch functions for independent streaming
async function fetchModelInferences(
  inference_id: string,
): Promise<ModelInferencesData> {
  const tensorZeroClient = getTensorZeroClient();
  const response = await tensorZeroClient.getModelInferences(inference_id);
  return resolveModelInferences(response.model_inferences);
}

async function fetchActionBarData(
  inference_id: string,
  functionName: string,
): Promise<ActionBarData> {
  const tensorZeroClient = getTensorZeroClient();
  const [demonstrationFeedback, usedVariants] = await Promise.all([
    tensorZeroClient.getDemonstrationFeedback(inference_id, { limit: 1 }),
    functionName === DEFAULT_FUNCTION
      ? tensorZeroClient.getUsedVariants(functionName)
      : Promise.resolve([]),
  ]);
  return {
    hasDemonstration: demonstrationFeedback.length > 0,
    usedVariants,
  };
}

async function fetchInput(inference: StoredInference): Promise<Input> {
  return loadFileDataForStoredInput(inference.input);
}

async function fetchFeedbackData(
  inference_id: string,
  params: {
    newFeedbackId: string | null;
    beforeFeedback: string | null;
    afterFeedback: string | null;
    limit: number;
  },
): Promise<FeedbackData> {
  const tensorZeroClient = getTensorZeroClient();
  const { newFeedbackId, beforeFeedback, afterFeedback, limit } = params;

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but eventually time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  if (newFeedbackId) {
    // Sequential case: poll first, then query bounds/metrics
    const feedback = await pollForFeedbackItem(
      inference_id,
      newFeedbackId,
      limit,
    );
    const [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ]);
    return { feedback, feedback_bounds, latestFeedbackByMetric };
  }

  // Normal case: execute all queries in parallel
  const [feedback, feedback_bounds, latestFeedbackByMetric] = await Promise.all(
    [
      tensorZeroClient.getFeedbackByTargetId(inference_id, {
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      }),
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ],
  );
  return { feedback, feedback_bounds, latestFeedbackByMetric };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const limit = Number(url.searchParams.get("limit")) || 10;
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");

  // Validate limit before deferring to ensure proper HTTP status
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  // Check inference exists before deferring to ensure 404 returns proper HTTP status
  const tensorZeroClient = getTensorZeroClient();
  const inferences = await tensorZeroClient.getInferences({
    ids: [inference_id],
    output_source: "inference",
  });
  if (inferences.inferences.length !== 1) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }

  const inference = inferences.inferences[0];

  // Return promises for independent streaming - each section loads as data becomes available
  return {
    inference,
    newFeedbackId,
    // Stream model inferences - used in BasicInfo header and Model Inferences table
    modelInferences: fetchModelInferences(inference_id),
    // Stream action bar data - hasDemonstration and usedVariants
    actionBarData: fetchActionBarData(inference_id, inference.function_name),
    // Stream input data
    input: fetchInput(inference),
    // Stream feedback data
    feedbackData: fetchFeedbackData(inference_id, {
      newFeedbackId,
      beforeFeedback,
      afterFeedback,
      limit,
    }),
  };
}

// Skeleton components
function ActionsSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-8" />
    </div>
  );
}

function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

function FeedbackTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Metric</TableHead>
        <TableHead>Value</TableHead>
        <TableHead>Tags</TableHead>
        <TableHead>Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

function FeedbackTableSkeleton() {
  return (
    <Table>
      <FeedbackTableHeaders />
      <TableBody>
        {Array.from({ length: 5 }).map((_, i) => (
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
  );
}

function ModelInferencesTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Model</TableHead>
        <TableHead>Input Tokens</TableHead>
        <TableHead>Output Tokens</TableHead>
        <TableHead>TTFT</TableHead>
        <TableHead>Response Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

function ModelInferencesTableSkeleton() {
  return (
    <Table>
      <ModelInferencesTableHeaders />
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

// Section error components
function FeedbackSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load feedback";

  return (
    <>
      <Table>
        <FeedbackTableHeaders />
        <TableBody>
          <TableRow>
            <TableCell colSpan={5}>
              <TableErrorNotice
                icon={AlertCircle}
                title="Error loading data"
                description={message}
              />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
      <PageButtons disabled />
    </>
  );
}

function ModelInferencesSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load model inferences";

  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        <TableRow>
          <TableCell colSpan={6}>
            <TableErrorNotice
              icon={AlertCircle}
              title="Error loading data"
              description={message}
            />
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}

function InputSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load input";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      {message}
    </div>
  );
}

// Content components for each streamed section
function BasicInfoContent({
  inference,
  modelInferences,
}: {
  inference: StoredInference;
  modelInferences: ModelInferencesData;
}) {
  const inferenceUsage = getTotalInferenceUsage(modelInferences);
  return (
    <BasicInfo
      inference={inference}
      inferenceUsage={inferenceUsage}
      modelInferences={modelInferences}
    />
  );
}

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

function prepareDemonstrationFromVariantOutput(
  variantOutput: VariantResponseInfo,
) {
  const output = variantOutput.output;
  if (output === undefined) {
    return undefined;
  }
  if (isJsonOutput(output)) {
    return output.parsed;
  } else {
    return output;
  }
}

/**
 * ActionBar component that handles TryWithSelect and VariantResponseModal.
 * This uses React.use() to resolve promises for input and modelInferences
 * so that the TryWithSelect can function properly.
 */
function InferenceActionBar({
  inference,
  actionBarData,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
}: {
  inference: StoredInference;
  actionBarData: ActionBarData;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}) {
  const { hasDemonstration, usedVariants } = actionBarData;
  const config = useConfig();
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([...usedVariants, ...config.model_names]);
  const models = [...modelsSet].sort();
  const options = isDefault ? models : variants;

  const { toast } = useToast();

  // Track resolved state of input and modelInferences
  const [resolvedInput, setResolvedInput] = useState<Input | null>(null);
  const [resolvedModelInferences, setResolvedModelInferences] =
    useState<ModelInferencesData | null>(null);

  // Resolve input promise
  useEffect(() => {
    inputPromise
      .then((input) => setResolvedInput(input))
      .catch(() => setResolvedInput(null));
  }, [inputPromise]);

  // Resolve modelInferences promise
  useEffect(() => {
    modelInferencesPromise
      .then((mi) => setResolvedModelInferences(mi))
      .catch(() => setResolvedModelInferences(null));
  }, [modelInferencesPromise]);

  const inferenceUsage = resolvedModelInferences
    ? getTotalInferenceUsage(resolvedModelInferences)
    : null;

  // Modal state
  const [openModal, setOpenModal] = useState<
    "human-feedback" | "variant-response" | null
  >(null);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

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
      onFeedbackAdded(fetcherData.redirectTo);
      setOpenModal(null);
      resetHumanFeedbackFetcher();
    }
  }, [
    humanFeedbackData,
    humanFeedbackState,
    onFeedbackAdded,
    resetHumanFeedbackFetcher,
  ]);

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
      onFeedbackAdded(fetcherData.redirectTo);
      setOpenModal(null);
      setSelectedVariant(null);
      resetDemonstrationFeedbackFetcher();
    }
  }, [
    demonstrationFeedbackData,
    demonstrationFeedbackState,
    onFeedbackAdded,
    resetDemonstrationFeedbackFetcher,
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
  const processRequest = useCallback(
    (
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
    },
    [submit, toast],
  );

  const onVariantSelect = useCallback(
    (variant: string) => {
      if (!resolvedInput) return;
      processRequest(variant, {
        resource: inference,
        input: resolvedInput,
        source: variantSource,
        variant,
      });
    },
    [inference, resolvedInput, processRequest],
  );

  const onModelSelect = useCallback(
    (model: string) => {
      if (!resolvedInput) return;
      processRequest(model, {
        resource: inference,
        input: resolvedInput,
        source: variantSource,
        model_name: model,
      });
    },
    [inference, resolvedInput, processRequest],
  );

  const handleRefresh = useCallback(() => {
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
  }, [lastRequestArgs, submit, toast]);

  const onSelect = isDefault ? onModelSelect : onVariantSelect;

  // Disable TryWithSelect until input is loaded (needed for making requests)
  const tryWithSelectDisabled = !resolvedInput;

  return (
    <>
      <ActionBar>
        <TryWithSelect
          options={options}
          onSelect={onSelect}
          isLoading={variantInferenceIsLoading}
          isDefaultFunction={isDefault}
          disabled={tryWithSelectDisabled}
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
              inferenceId={inference.inference_id}
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

      {selectedVariant && inferenceUsage && (
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
    </>
  );
}

function FeedbackSectionContent({ data }: { data: FeedbackData }) {
  const { feedback, feedback_bounds, latestFeedbackByMetric } = data;
  const navigate = useNavigate();

  const topFeedback = feedback[0] as { id: string } | undefined;
  const bottomFeedback = feedback[feedback.length - 1] as
    | { id: string }
    | undefined;

  const handleNextPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.delete("newFeedbackId");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.delete("newFeedbackId");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious =
    !topFeedback?.id ||
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNext =
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
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={disablePrevious}
        disableNext={disableNext}
      />
    </>
  );
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    newFeedbackId,
    modelInferences,
    actionBarData,
    input,
    feedbackData,
  } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();

  // Show toast when feedback is successfully added (outside Suspense to avoid repeating)
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  // Handle feedback added callback - extract newFeedbackId from the API redirect URL
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

  return (
    <PageLayout>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Inferences", href: "/observability/inferences" },
            ]}
          />
        }
        name={inference.inference_id}
      >
        {/* BasicInfo - streams independently when modelInferences resolves */}
        <Suspense
          key={location.key}
          fallback={<BasicInfoLayoutSkeleton rows={5} />}
        >
          <Await
            resolve={modelInferences}
            errorElement={<BasicInfoLayoutSkeleton rows={5} />}
          >
            {(resolvedModelInferences) => (
              <BasicInfoContent
                inference={inference}
                modelInferences={resolvedModelInferences}
              />
            )}
          </Await>
        </Suspense>

        {/* ActionBar - streams when actionBarData resolves */}
        <Suspense
          key={`actions-${location.key}`}
          fallback={<ActionsSkeleton />}
        >
          <Await resolve={actionBarData} errorElement={<ActionsSkeleton />}>
            {(resolvedActionBarData) => (
              <InferenceActionBar
                inference={inference}
                actionBarData={resolvedActionBarData}
                inputPromise={input}
                modelInferencesPromise={modelInferences}
                onFeedbackAdded={handleFeedbackAdded}
              />
            )}
          </Await>
        </Suspense>
      </PageHeader>

      <SectionsGroup>
        {/* Input section - streams independently */}
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Suspense key={`input-${location.key}`} fallback={<InputSkeleton />}>
            <Await resolve={input} errorElement={<InputSectionError />}>
              {(resolvedInput) => <InputElement input={resolvedInput} />}
            </Await>
          </Suspense>
        </SectionLayout>

        {/* Output section - renders immediately since inference is sync */}
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

        {/* Feedback section - streams independently */}
        <SectionLayout>
          <SectionHeader
            heading="Feedback"
            badge={{
              name: "inference",
              tooltip:
                "This table only includes inference-level feedback. To see episode-level feedback, open the detail page for that episode.",
            }}
          />
          <Suspense
            key={`feedback-${location.key}`}
            fallback={
              <>
                <FeedbackTableSkeleton />
                <PageButtons disabled />
              </>
            }
          >
            <Await
              resolve={feedbackData}
              errorElement={<FeedbackSectionError />}
            >
              {(resolvedFeedbackData) => (
                <FeedbackSectionContent data={resolvedFeedbackData} />
              )}
            </Await>
          </Suspense>
        </SectionLayout>

        {/* Inference Parameters - renders immediately */}
        <SectionLayout>
          <SectionHeader heading="Inference Parameters" />
          <ParameterCard
            parameters={JSON.stringify(inference.inference_params, null, 2)}
          />
        </SectionLayout>

        {/* Tool Parameters - renders immediately for chat type */}
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

        {/* Tags - renders immediately */}
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

        {/* Model Inferences table - streams when modelInferences resolves */}
        <SectionLayout>
          <SectionHeader heading="Model Inferences" />
          <Suspense
            key={`model-inferences-${location.key}`}
            fallback={<ModelInferencesTableSkeleton />}
          >
            <Await
              resolve={modelInferences}
              errorElement={<ModelInferencesSectionError />}
            >
              {(resolvedModelInferences) => (
                <ModelInferencesTable
                  modelInferences={resolvedModelInferences}
                />
              )}
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
