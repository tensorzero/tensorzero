import { Suspense } from "react";
import { Await, useAsyncError } from "react-router";
import type { StoredInference } from "~/types/tensorzero";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { useFunctionConfig } from "~/context/config";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import {
  BasicInfoLayout,
  BasicInfoLayoutSkeleton,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import {
  Timer,
  Calendar,
  InputIcon,
  Output,
  Cached,
} from "~/components/icons/Icons";
import { toFunctionUrl, toVariantUrl, toEpisodeUrl } from "~/utils/urls";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { ModelInferencesData } from "./inference-data.server";

// Streaming wrapper with Suspense/Await (lives in PageHeader, not SectionsGroup)
interface BasicInfoStreamingProps {
  inference: StoredInference;
  promise: Promise<ModelInferencesData>;
  locationKey: string;
}

export function BasicInfoStreaming({
  inference,
  promise,
  locationKey,
}: BasicInfoStreamingProps) {
  return (
    <Suspense key={locationKey} fallback={<BasicInfoSkeleton />}>
      <Await resolve={promise} errorElement={<BasicInfoError />}>
        {(modelInferences) => (
          <BasicInfoContent
            inference={inference}
            modelInferences={modelInferences}
          />
        )}
      </Await>
    </Suspense>
  );
}

// Content
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

// Also exported for use by InferenceDetailContent (non-streaming version)
interface BasicInfoProps {
  inference: StoredInference;
  inferenceUsage?: InferenceUsage;
  modelInferences?: ParsedModelInferenceRow[];
}

export function BasicInfo({
  inference,
  inferenceUsage,
  modelInferences = [],
}: BasicInfoProps) {
  const functionConfig = useFunctionConfig(inference.function_name);
  const variantType =
    functionConfig?.variants[inference.variant_name]?.inner.type ??
    (inference.function_name === "tensorzero::default"
      ? "chat_completion"
      : "unknown");

  const timestampTooltip = <TimestampTooltip timestamp={inference.timestamp} />;
  const functionIconConfig = getFunctionTypeIcon(inference.type);

  const hasCachedInferences = modelInferences.some((mi) => mi.cached);
  const allCached =
    modelInferences.length > 0 && modelInferences.every((mi) => mi.cached);
  const cacheStatus = allCached
    ? "FULL"
    : hasCachedInferences
      ? "PARTIAL"
      : "NONE";

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={inference.function_name}
            secondaryLabel={`· ${inference.type}`}
            link={toFunctionUrl(inference.function_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.variant_name}
            secondaryLabel={`· ${variantType}`}
            link={toVariantUrl(inference.function_name, inference.variant_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.episode_id}
            link={toEpisodeUrl(inference.episode_id)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<InputIcon className="text-fg-tertiary" />}
            label={`${inferenceUsage?.input_tokens ?? ""} tok`}
            tooltip="Input Tokens"
          />
          <Chip
            icon={<Output className="text-fg-tertiary" />}
            label={`${inferenceUsage?.output_tokens ?? ""} tok`}
            tooltip="Output Tokens"
          />
          <Chip
            icon={<Timer className="text-fg-tertiary" />}
            label={`${inference.processing_time_ms} ms`}
            tooltip="Processing Time"
          />
          {(cacheStatus === "FULL" || cacheStatus === "PARTIAL") && (
            <Chip
              icon={<Cached className="text-fg-tertiary" />}
              label={cacheStatus === "FULL" ? "Cached" : "Partially Cached"}
              tooltip={
                cacheStatus === "FULL"
                  ? "All model inferences were cached by TensorZero"
                  : "Some model inferences were cached by TensorZero"
              }
            />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(new Date(inference.timestamp))}
            tooltip={timestampTooltip}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}

// Skeleton
function BasicInfoSkeleton() {
  return <BasicInfoLayoutSkeleton rows={5} />;
}

// Error
function BasicInfoError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load inference details";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  );
}
