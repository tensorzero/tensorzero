import { Suspense, useEffect, useState, useCallback } from "react";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { Route } from "./+types/route";
import {
  Await,
  data,
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
import type { StoredInference } from "~/types/tensorzero";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "./InferenceParameters";
import { ToolParametersSection } from "~/components/inference/ToolParametersSection";
import { TagsTable } from "~/components/tags/TagsTable";
import { ModelInferencesTable } from "./ModelInferencesTable";
import BasicInfo from "./InferenceBasicInfo";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";

// Local imports
import {
  fetchModelInferences,
  fetchActionBarData,
  fetchInput,
  fetchFeedbackData,
  type ModelInferencesData,
  type FeedbackData,
} from "./inference-data.server";
import {
  ActionsSkeleton,
  InputSkeleton,
  FeedbackTableSkeleton,
  ModelInferencesTableSkeleton,
} from "./InferenceSkeletons";
import {
  FeedbackSectionError,
  ModelInferencesSectionError,
  InputSectionError,
} from "./InferenceSectionErrors";
import { InferenceActionBar } from "./InferenceActionBar";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

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

function FeedbackSectionContent({
  data,
  onCountUpdate,
}: {
  data: FeedbackData;
  onCountUpdate: (count: number) => void;
}) {
  const { feedback, feedback_bounds, latestFeedbackByMetric } = data;
  const navigate = useNavigate();

  // Update count when data loads
  useEffect(() => {
    onCountUpdate(feedback.length);
  }, [feedback.length, onCountUpdate]);

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

  // Track feedback count for SectionHeader (updated when feedbackData resolves)
  const [feedbackCount, setFeedbackCount] = useState<number | undefined>(
    undefined,
  );

  // Show toast when feedback is successfully added (outside Suspense to avoid repeating)
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  // Reset feedback count when location changes (navigating to new page)
  useEffect(() => {
    setFeedbackCount(undefined);
  }, [location.key]);

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
            count={feedbackCount}
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
                <FeedbackSectionContent
                  data={resolvedFeedbackData}
                  onCountUpdate={setFeedbackCount}
                />
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
