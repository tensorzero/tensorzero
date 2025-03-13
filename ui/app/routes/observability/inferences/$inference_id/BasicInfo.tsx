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
import {
  EpisodeChip,
  FunctionChip,
  VariantChip,
  TimestampChip,
  ProcessingTimeChip,
} from "~/components/ui/Chip";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";

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
  return (
    <div className="space-y-4">
      <BasicInfoLayout>
        <BasicInfoItem>
          <BasicInfoItemTitle>Function</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <FunctionChip
              name={inference.function_name}
              link={`/observability/functions/${inference.function_name}`}
              type={inference.function_type}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Variant</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <VariantChip
              name={inference.variant_name}
              link={`/observability/functions/${inference.function_name}/variants/${inference.variant_name}`}
              type={variantType}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <EpisodeChip
              text={inference.episode_id}
              link={`/observability/episodes/${inference.episode_id}`}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <div className="flex flex-row gap-4">
              <div className="flex flex-row gap-1">
                <span className="text-foreground-secondary">Input </span>
                <span className="text-foreground-primary">
                  {inferenceUsage?.input_tokens ?? ""}
                </span>
              </div>
              <div className="flex flex-row gap-1">
                <span className="text-foreground-secondary">Output </span>
                <span className="text-foreground-primary">
                  {inferenceUsage?.output_tokens ?? ""}
                </span>
              </div>
            </div>
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <TimestampChip timestamp={inference.timestamp} />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Processing Time</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <ProcessingTimeChip
              processingTimeMs={inference.processing_time_ms}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Tags</BasicInfoItemTitle>
          <BasicInfoItemContent>
            {Object.keys(inference.tags).length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(inference.tags).map(([key, value]) => (
                  <div key={key} className="flex items-center gap-1">
                    <span className="text-foreground-secondary">{key}:</span>
                    <span className="text-foreground-primary">{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="text-foreground-secondary">No tags</span>
            )}
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
