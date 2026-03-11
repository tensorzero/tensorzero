import type { VariantPerformanceRow } from "~/types/tensorzero";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { TimeGranularitySelector } from "./TimeGranularitySelector";
import { useTimeGranularityParam } from "~/hooks/use-time-granularity-param";
import {
  VariantPerformanceChart,
  transformVariantPerformances,
} from "./VariantPerformanceChart";

export function VariantPerformance({
  variant_performances,
  metric_name,
  singleVariantMode = false,
}: {
  variant_performances: VariantPerformanceRow[];
  metric_name: string;
  singleVariantMode?: boolean;
}) {
  const [time_granularity, onTimeGranularityChange] = useTimeGranularityParam(
    "time_granularity",
    "week",
  );
  const { data, variantNames } =
    transformVariantPerformances(variant_performances);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between">
          <div>
            <CardTitle>
              {singleVariantMode
                ? "Performance Over Time"
                : "Variant Performance Over Time"}
            </CardTitle>
            <CardDescription>
              {singleVariantMode ? (
                <span>
                  Showing average metric values for <code>{metric_name}</code>
                </span>
              ) : (
                <span>
                  Showing average metric values by variant for metric{" "}
                  <code>{metric_name}</code>
                </span>
              )}
            </CardDescription>
          </div>
          <TimeGranularitySelector
            time_granularity={time_granularity}
            onTimeGranularityChange={onTimeGranularityChange}
          />
        </CardHeader>
        <CardContent>
          <VariantPerformanceChart
            data={data}
            variantNames={variantNames}
            timeGranularity={time_granularity}
            singleVariantMode={singleVariantMode}
          />
        </CardContent>
      </Card>
    </div>
  );
}
