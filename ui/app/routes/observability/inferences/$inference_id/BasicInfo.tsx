import { Suspense } from "react";
import { Await } from "react-router";
import type { StoredInference } from "~/types/tensorzero";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
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
  Cost,
} from "~/components/icons/Icons";
import { toFunctionUrl, toVariantUrl, toEpisodeUrl } from "~/utils/urls";
import { formatCost } from "~/utils/cost";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { InlineAsyncError } from "~/components/ui/error/ErrorContentPrimitives";
import type { ModelInferencesData } from "./inference-data.server";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";

function formatCacheTooltip(usage: InferenceUsage | undefined): string {
  const parts: string[] = [];
  if (usage?.provider_cache_read_input_tokens != null) {
    parts.push(`Cache read: ${usage.provider_cache_read_input_tokens} tokens`);
  }
  if (usage?.provider_cache_write_input_tokens != null) {
    parts.push(
      `Cache write: ${usage.provider_cache_write_input_tokens} tokens`,
    );
  }
  return parts.join("\n");
}

interface BasicInfoStreamingProps {
  inference: StoredInference;
  variantType: string;
  promise: Promise<ModelInferencesData>;
  locationKey: string;
}

export function BasicInfoStreaming({
  inference,
  variantType,
  promise,
  locationKey,
}: BasicInfoStreamingProps) {
  return (
    <Suspense key={locationKey} fallback={<BasicInfoLayoutSkeleton rows={5} />}>
      <Await
        resolve={promise}
        errorElement={
          <InlineAsyncError defaultMessage="Failed to load inference details" />
        }
      >
        {(modelInferences) => (
          <BasicInfo
            inference={inference}
            variantType={variantType}
            modelInferences={modelInferences}
          />
        )}
      </Await>
    </Suspense>
  );
}

interface BasicInfoProps {
  inference: StoredInference;
  variantType: string;
  modelInferences?: ParsedModelInferenceRow[];
}

export function BasicInfo({
  inference,
  variantType,
  modelInferences = [],
}: BasicInfoProps) {
  const snapshotHash = inference.snapshot_hash;
  const inferenceUsage = getTotalInferenceUsage(modelInferences);

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
            link={toFunctionUrl(inference.function_name, snapshotHash)}
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
            link={toVariantUrl(
              inference.function_name,
              inference.variant_name,
              snapshotHash,
            )}
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
            secondaryLabel={
              inferenceUsage?.provider_cache_read_input_tokens != null &&
              inferenceUsage.provider_cache_read_input_tokens > 0
                ? `(${inferenceUsage.provider_cache_read_input_tokens} cached)`
                : undefined
            }
            tooltip={formatCacheTooltip(inferenceUsage) || "Input Tokens"}
          />
          <Chip
            icon={<Output className="text-fg-tertiary" />}
            label={`${inferenceUsage?.output_tokens ?? ""} tok`}
            tooltip="Output Tokens"
          />
          {inferenceUsage?.cost != null && (
            <Chip
              icon={<Cost className="text-fg-tertiary" />}
              label={formatCost(inferenceUsage.cost)}
              tooltip="Cost"
            />
          )}
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
            tooltip={<TimestampTooltip timestamp={inference.timestamp} />}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
