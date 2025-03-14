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
  Functions,
  SupervisedFineTuning,
  Episodes,
  Placeholder,
} from "~/components/icons/Icons";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";

// Helper function for formatting dates
const formatDate = (date: Date) => {
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    second: "numeric",
    hour12: true,
  };

  return new Date(date).toLocaleString("en-US", options);
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
  return (
    <div className="space-y-4">
      <BasicInfoLayout>
        <BasicInfoItem>
          <BasicInfoItemTitle>Function</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              icon={<Functions />}
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
              icon={<SupervisedFineTuning />}
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
              icon={<Episodes className="text-foreground-tertiary" />}
              label={inference.episode_id}
              link={`/observability/episodes/${inference.episode_id}`}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <div className="flex flex-row gap-2">
              <Chip
                icon={<Placeholder />}
                label={`${inferenceUsage?.input_tokens ?? ""} tok`}
              />
              <Chip
                icon={<Placeholder />}
                label={`${inferenceUsage?.output_tokens ?? ""} tok`}
              />
            </div>
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              icon={<Placeholder className="text-foreground-tertiary" />}
              label={formatDate(new Date(inference.timestamp))}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>

        <BasicInfoItem>
          <BasicInfoItemTitle>Processing Time</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              icon={<Placeholder className="text-foreground-tertiary" />}
              label={`${inference.processing_time_ms} ms`}
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
