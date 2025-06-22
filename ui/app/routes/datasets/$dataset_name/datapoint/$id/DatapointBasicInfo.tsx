import { useConfig } from "~/context/config";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Calendar, Dataset } from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import { getFunctionTypeIcon } from "~/utils/icon";

interface BasicInfoProps {
  datapoint: ParsedDatasetRow;
}

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

export default function BasicInfo({ datapoint }: BasicInfoProps) {
  const config = useConfig();
  const function_config = config.functions[datapoint.function_name];
  const type = function_config?.type;
  if (!type) {
    throw new Error(`Function ${datapoint.function_name} not found`);
  }

  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(datapoint.updated_at);

  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(type);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Dataset</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Dataset className="text-fg-tertiary" />}
            label={datapoint.dataset_name}
            link={`/datasets/${datapoint.dataset_name}`}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={datapoint.function_name}
            secondaryLabel={type}
            link={`/observability/functions/${datapoint.function_name}`}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Inference</BasicInfoItemTitle>
        <BasicInfoItemContent>
          {datapoint.source_inference_id ? (
            <Chip
              label={datapoint.source_inference_id}
              link={`/observability/inferences/${datapoint.source_inference_id}`}
              font="mono"
            />
          ) : (
            <Chip label="Edited from original" prominence="muted" />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
        <BasicInfoItemContent>
          {datapoint.episode_id ? (
            <Chip
              label={datapoint.episode_id}
              link={`/observability/episodes/${datapoint.episode_id}`}
              font="mono"
            />
          ) : (
            <Chip label="None" prominence="muted" />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(new Date(datapoint.updated_at))}
            tooltip={timestampTooltip}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
