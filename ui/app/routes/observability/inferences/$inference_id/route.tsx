import { useEffect, useState, useCallback } from "react";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getConfigForSnapshot } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import {
  data,
  useLocation,
  useNavigate,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
} from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { getPageErrorInfo } from "~/utils/tensorzero/errors";
import { AlertTriangle } from "lucide-react";
import { useToast } from "~/hooks/use-toast";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { ParameterCard } from "./ParameterCard";
import { ToolParametersSection } from "~/components/inference/ToolParametersSection";
import { TagsTable } from "~/components/tags/TagsTable";
import {
  fetchModelInferences,
  fetchUsedVariants,
  fetchHasDemonstration,
  fetchInput,
  fetchFeedbackData,
} from "./inference-data.server";
import { BasicInfoStreaming } from "./BasicInfo";
import { InferenceActionBar } from "./InferenceActionBar";
import { InputSection } from "./InputSection";
import { FeedbackSection } from "./FeedbackSection";
import { ModelInferencesSection } from "./ModelInferencesSection";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

/**
 * Prevent revalidation when fetchers submit to API routes.
 * With streaming/deferred data, revalidation re-runs the loader and waits for
 * deferred promises to resolve, keeping the fetcher in "loading" state and
 * blocking downstream effects that depend on "idle" state.
 */
export function shouldRevalidate({
  formAction,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  if (
    formAction?.startsWith("/api/feedback") ||
    formAction?.startsWith("/api/datasets/datapoints/from-inference") ||
    formAction?.startsWith("/api/tensorzero/inference")
  ) {
    return false;
  }
  return defaultShouldRevalidate;
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const limit = Number(url.searchParams.get("limit")) || 10;
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");

  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const tensorZeroClient = getTensorZeroClient();
  const inferences = await tensorZeroClient.getInferences({
    ids: [inference_id],
    output_source: "inference",
  });
  if (inferences.inferences.length !== 1) {
    throw data(`No inference found for id ${inference_id}.`, { status: 404 });
  }

  const inference = inferences.inferences[0];

  const snapshotConfig = await getConfigForSnapshot(inference.snapshot_hash);

  const snapshotFunctionConfig =
    snapshotConfig.functions[inference.function_name];
  const variantType =
    snapshotFunctionConfig?.variants[inference.variant_name]?.inner.type ??
    (inference.function_name === "tensorzero::default"
      ? "chat_completion"
      : "unknown");

  return {
    inference,
    variantType,
    newFeedbackId,
    modelInferences: fetchModelInferences(inference_id),
    usedVariants: fetchUsedVariants(inference.function_name),
    hasDemonstration: fetchHasDemonstration(inference_id),
    input: fetchInput(inference),
    feedbackData: fetchFeedbackData(inference_id, {
      newFeedbackId,
      beforeFeedback,
      afterFeedback,
      limit,
    }),
  };
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    variantType,
    newFeedbackId,
    modelInferences,
    usedVariants,
    hasDemonstration,
    input,
    feedbackData,
  } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [feedbackCount, setFeedbackCount] = useState<number | undefined>(
    undefined,
  );

  // Show toast when feedback is added
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  // Reset feedback count on navigation
  useEffect(() => {
    setFeedbackCount(undefined);
  }, [location.key]);

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
        <BasicInfoStreaming
          inference={inference}
          variantType={variantType}
          promise={modelInferences}
          locationKey={location.key}
        />
        <InferenceActionBar
          inference={inference}
          usedVariantsPromise={usedVariants}
          hasDemonstrationPromise={hasDemonstration}
          inputPromise={input}
          modelInferencesPromise={modelInferences}
          onFeedbackAdded={handleFeedbackAdded}
          locationKey={location.key}
        />
      </PageHeader>

      <SectionsGroup>
        <InputSection promise={input} locationKey={location.key} />

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

        <FeedbackSection
          promise={feedbackData}
          locationKey={location.key}
          count={feedbackCount}
          onCountUpdate={setFeedbackCount}
        />

        <SectionLayout>
          <SectionHeader heading="Inference Parameters" />
          {inference.inference_params ? (
            <ParameterCard
              parameters={JSON.stringify(inference.inference_params, null, 2)}
            />
          ) : (
            <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
              Parameters missing
            </div>
          )}
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

        <ModelInferencesSection
          promise={modelInferences}
          locationKey={location.key}
        />
      </SectionsGroup>
    </PageLayout>
  );
}

export function ErrorBoundary({ params, error }: Route.ErrorBoundaryProps) {
  const { title, message, status } = getPageErrorInfo(error);

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
        name={params.inference_id}
      />
      <SectionsGroup>
        <SectionErrorNotice
          icon={AlertTriangle}
          title={status ? `Error ${status}` : title}
          description={message}
        />
      </SectionsGroup>
    </PageLayout>
  );
}
