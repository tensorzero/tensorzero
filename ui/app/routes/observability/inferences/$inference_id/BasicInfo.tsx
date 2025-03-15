import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { useConfig } from "~/context/config";
import {
  type TryWithVariantButtonProps,
  TryWithVariantButton,
} from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import {
  TypeChat,
  TypeJson,
  Timer,
  Calendar,
  Input,
  Output,
} from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";

// Helper function to get the appropriate icon and background based on function type
const getFunctionIcon = (functionType: string) => {
  switch (functionType?.toLowerCase()) {
    default:
      return {
        icon: <TypeJson className="text-fg-type-json" />,
        iconBg: "bg-bg-type-json",
      };
    case "chat":
    case "conversation":
      return {
        icon: <TypeChat className="text-fg-type-chat" />,
        iconBg: "bg-bg-type-chat",
      };
  }
};

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
  tryWithVariantProps: TryWithVariantButtonProps;
  dataset_counts: DatasetCountInfo[];
  onDatasetSelect: (
    dataset: string,
    output: "inference" | "demonstration" | "none",
  ) => void;
  hasDemonstration: boolean;
}

export default function BasicInfo({
  inference,
  inferenceUsage,
  tryWithVariantProps,
  dataset_counts,
  onDatasetSelect,
  hasDemonstration,
}: BasicInfoProps) {
  const config = useConfig();
  const variantType =
    config.functions[inference.function_name]?.variants[inference.variant_name]
      ?.type;

  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(inference.timestamp);

  // Get function icon and background
  const functionIconConfig = getFunctionIcon(inference.function_type);

  return (
    <div className="space-y-8">
      <BasicInfoLayout>
        <BasicInfoItem>
          <BasicInfoItemTitle>Function</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              icon={functionIconConfig.icon}
              iconBg={functionIconConfig.iconBg}
              label={inference.function_name}
              secondaryLabel={inference.function_type}
              link={`/observability/functions/${inference.function_name}`}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Variant</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={inference.variant_name}
              secondaryLabel={variantType}
              link={`/observability/functions/${inference.function_name}/variants/${inference.variant_name}`}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={inference.episode_id}
              link={`/observability/episodes/${inference.episode_id}`}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <div className="flex flex-row gap-1">
              <Chip
                icon={<Input className="text-fg-tertiary" />}
                label={`${inferenceUsage?.input_tokens ?? ""} tok`}
                tooltip="Input Tokens"
              />
              <Chip
                icon={<Output className="text-fg-tertiary" />}
                label={`${inferenceUsage?.output_tokens ?? ""} tok`}
                tooltip="Output Tokens"
              />
            </div>
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

        <BasicInfoItem>
          <BasicInfoItemTitle>Processing Time</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              icon={<Timer className="text-fg-tertiary" />}
              label={`${inference.processing_time_ms} ms`}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      </BasicInfoLayout>

      <div className="flex gap-2">
        <TryWithVariantButton {...tryWithVariantProps} />
        {FF_ENABLE_DATASETS && (
          <AddToDatasetButton
            dataset_counts={dataset_counts}
            onDatasetSelect={onDatasetSelect}
            hasDemonstration={hasDemonstration}
          />
        )}
      </div>
    </div>
  );
}
