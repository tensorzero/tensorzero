import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Calendar } from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import type {
  DynamicEvaluationRun,
  DynamicEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/dynamic_evaluations";
import { Link } from "react-router";

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
  dynamicEvaluationRun: DynamicEvaluationRun;
  count: number;
  statistics: DynamicEvaluationRunStatisticsByMetricName[];
}

export default function BasicInfo({
  dynamicEvaluationRun,
  count,
  statistics,
}: BasicInfoProps) {
  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(
    dynamicEvaluationRun.timestamp,
  );

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        {dynamicEvaluationRun.name ? (
          <>
            <BasicInfoItemTitle>Name</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={dynamicEvaluationRun.name} font="mono" />
            </BasicInfoItemContent>
          </>
        ) : null}
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>ID</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={dynamicEvaluationRun.id} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Project</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={dynamicEvaluationRun.project_name} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant Pins</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex flex-wrap gap-1">
            {Object.entries(dynamicEvaluationRun.variant_pins).map(([k, v]) => (
              <div
                key={k}
                className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs"
              >
                <Link
                  to={`/observability/functions/${k}`}
                  className="text-blue-600 hover:underline"
                >
                  {k}
                </Link>
                <span>:</span>
                <Link
                  to={`/observability/functions/${k}/variants/${v}`}
                  className="text-blue-600 hover:underline"
                >
                  {v}
                </Link>
              </div>
            ))}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Tags</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex flex-wrap gap-1">
            {Object.entries(dynamicEvaluationRun.tags).map(([k, v]) => (
              <Chip key={k} label={`${k}: ${v}`} font="mono" />
            ))}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Episodes</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={`${count}`} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Statistics</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex flex-col gap-1">
            {statistics.map((stat) => (
              <div key={stat.metric_name}>
                <span className="font-semibold">{stat.metric_name}: </span>
                <span>
                  {stat.avg_metric.toFixed(3)} ± {stat.ci_error.toFixed(3)} (n=
                  {stat.count})
                </span>
              </div>
            ))}
            {statistics.length === 0 && (
              <span className="text-xs text-gray-500">
                No statistics available.
              </span>
            )}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(
              new Date(dynamicEvaluationRun.timestamp),
            )}
            tooltip={timestampTooltip}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
