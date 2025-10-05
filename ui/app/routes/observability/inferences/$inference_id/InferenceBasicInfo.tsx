import type {
  ParsedInferenceRow,
  ParsedModelInferenceRow,
} from "~/utils/clickhouse/inference";
import { useFunctionConfig } from "~/context/config";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
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

interface BasicInfoProps {
  inference: ParsedInferenceRow;
  inferenceUsage?: InferenceUsage;
  modelInferences?: ParsedModelInferenceRow[];
}

export default function BasicInfo({
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

  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(inference.timestamp);

  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(inference.function_type);

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
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={inference.function_name}
            secondaryLabel={`· ${inference.function_type}`}
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
