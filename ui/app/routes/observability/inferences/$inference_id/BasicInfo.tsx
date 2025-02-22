import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import { Badge } from "~/components/ui/badge";
import { Link } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { useConfig } from "~/context/config";
import {
  type TryWithVariantButtonProps,
  TryWithVariantButton,
} from "~/components/inference/TryWithVariantButton";
import { AddToDatasetButton } from "./AddToDatasetButton";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";

interface BasicInfoProps {
  inference: ParsedInferenceRow;
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
          <AddToDatasetButton
            dataset_counts={dataset_counts}
            onDatasetSelect={onDatasetSelect}
            hasDemonstration={hasDemonstration}
          />
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
            <Badge variant="outline" className="bg-blue-200">
              {variantType}
            </Badge>
          </div>
          <div>
            <dt className="text-lg font-semibold">Episode ID</dt>
            <dd>
              <Link to={`/observability/episodes/${inference.episode_id}`}>
                <Code>{inference.episode_id}</Code>
              </Link>
            </dd>
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
