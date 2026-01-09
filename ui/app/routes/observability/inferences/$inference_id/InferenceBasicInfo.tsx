import { Suspense } from "react";
import { Await } from "react-router";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { useFunctionConfig } from "~/context/config";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
  BasicInfoLayout,
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
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { StoredInference } from "~/types/tensorzero";

// Create timestamp tooltip component
const createTimestampTooltip = (timestamp: string | number | Date) => {
  const { formattedDate, formattedTime, relativeTime } =
    getTimestampTooltipData(timestamp);

  return (
    <div className="flex flex-col gap-1">
      <div>{formattedDate}</div>
      <div>{formattedTime}</div>
      <div>{relativeTime}</div>
    </div>
  );
};

// Support both server-side streaming (promise) and client-side resolved data
type BasicInfoProps = {
  inference: StoredInference;
} & (
  | {
      modelInferencesPromise: Promise<ParsedModelInferenceRow[]>;
      modelInferences?: never;
    }
  | {
      modelInferences: ParsedModelInferenceRow[];
      modelInferencesPromise?: never;
    }
  | { modelInferences?: never; modelInferencesPromise?: never }
);

// Convert bigint processing_time_ms to number for display
function getProcessingTimeMs(inference: StoredInference): number | null {
  if (inference.processing_time_ms === undefined) return null;
  return typeof inference.processing_time_ms === "bigint"
    ? Number(inference.processing_time_ms)
    : inference.processing_time_ms;
}

// Inner component for usage chips (rendered inside Suspense)
function UsageChips({
  modelInferences,
  processingTimeMs,
}: {
  modelInferences: ParsedModelInferenceRow[];
  processingTimeMs: number | null;
}) {
  const inferenceUsage = getTotalInferenceUsage(modelInferences);

  // Determine cache status from model inferences
  const hasCachedInferences = modelInferences.some((mi) => mi.cached);
  const allCached =
    modelInferences.length > 0 && modelInferences.every((mi) => mi.cached);
  const cacheStatus = allCached
    ? "FULL"
    : hasCachedInferences
      ? "PARTIAL"
      : "NONE";

  return (
    <>
      <Chip
        icon={<InputIcon className="text-fg-tertiary" />}
        label={`${inferenceUsage.input_tokens ?? "—"} tok`}
        tooltip="Input Tokens"
      />
      <Chip
        icon={<Output className="text-fg-tertiary" />}
        label={`${inferenceUsage.output_tokens ?? "—"} tok`}
        tooltip="Output Tokens"
      />
      <Chip
        icon={<Timer className="text-fg-tertiary" />}
        label={`${processingTimeMs ?? "—"} ms`}
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
    </>
  );
}

// Loading state for usage chips
function UsageChipsLoading({
  processingTimeMs,
}: {
  processingTimeMs: number | null;
}) {
  return (
    <>
      <Chip
        icon={<InputIcon className="text-fg-tertiary" />}
        label="— tok"
        tooltip="Input Tokens"
      />
      <Chip
        icon={<Output className="text-fg-tertiary" />}
        label="— tok"
        tooltip="Output Tokens"
      />
      <Chip
        icon={<Timer className="text-fg-tertiary" />}
        label={`${processingTimeMs ?? "—"} ms`}
        tooltip="Processing Time"
      />
    </>
  );
}

export default function BasicInfo(props: BasicInfoProps) {
  const { inference } = props;
  const functionConfig = useFunctionConfig(inference.function_name);
  const variantType =
    functionConfig?.variants[inference.variant_name]?.inner.type ??
    (inference.function_name === "tensorzero::default"
      ? "chat_completion"
      : "unknown");

  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(inference.timestamp);

  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(inference.type);

  // Get processing time as number
  const processingTimeMs = getProcessingTimeMs(inference);

  // Determine which usage display to render
  const renderUsageContent = () => {
    // Case 1: Server-side streaming with promise
    if ("modelInferencesPromise" in props && props.modelInferencesPromise) {
      return (
        <Suspense
          fallback={<UsageChipsLoading processingTimeMs={processingTimeMs} />}
        >
          <Await resolve={props.modelInferencesPromise}>
            {(modelInferences) => (
              <UsageChips
                modelInferences={modelInferences}
                processingTimeMs={processingTimeMs}
              />
            )}
          </Await>
        </Suspense>
      );
    }

    // Case 2: Client-side with resolved data
    if ("modelInferences" in props && props.modelInferences) {
      return (
        <UsageChips
          modelInferences={props.modelInferences}
          processingTimeMs={processingTimeMs}
        />
      );
    }

    // Case 3: No data available yet
    return <UsageChipsLoading processingTimeMs={processingTimeMs} />;
  };

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
        <BasicInfoItemContent>{renderUsageContent()}</BasicInfoItemContent>
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
