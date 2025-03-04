import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import { Link } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { useConfig } from "~/context/config";
import {
  type TryWithVariantButtonProps,
  TryWithVariantButton,
} from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";

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
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-xl">Basic Information</CardTitle>
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
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="text-lg font-semibold">Function</dt>
            <dd>
              <Link to={`/observability/functions/${inference.function_name}`}>
                <Code>{inference.function_name}</Code>
              </Link>
            </dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Variant</dt>
            <dd>
              <Link
                to={`/observability/functions/${inference.function_name}/variants/${inference.variant_name}`}
              >
                <Code>{inference.variant_name}</Code>
              </Link>
            </dd>
            <Code>{variantType}</Code>
          </div>
          <div className="col-span-2">
            <dt className="text-lg font-semibold">Episode ID</dt>
            <dd>
              <Link to={`/observability/episodes/${inference.episode_id}`}>
                <Code>{inference.episode_id}</Code>
              </Link>
            </dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Input Tokens</dt>
            <dd>{inferenceUsage?.input_tokens ?? ""}</dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Output Tokens</dt>
            <dd>{inferenceUsage?.output_tokens ?? ""}</dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Timestamp</dt>
            <dd>{new Date(inference.timestamp).toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Processing Time</dt>
            <dd>{inference.processing_time_ms}ms</dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
