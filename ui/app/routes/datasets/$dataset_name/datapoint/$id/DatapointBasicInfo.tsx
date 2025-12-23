import { useFunctionConfig } from "~/context/config";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import EditableChip from "~/components/ui/EditableChip";
import { Calendar, Dataset } from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import { getFunctionTypeIcon } from "~/utils/icon";
import {
  toDatasetUrl,
  toFunctionUrl,
  toInferenceUrl,
  toEpisodeUrl,
} from "~/utils/urls";
import { useReadOnly } from "~/context/read-only";
import type { Datapoint } from "~/types/tensorzero";

interface BasicInfoProps {
  datapoint: Datapoint;
  onRenameDatapoint?: (newName: string) => void | Promise<void>;
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

export default function DatapointBasicInfo({
  datapoint,
  onRenameDatapoint,
}: BasicInfoProps) {
  const function_config = useFunctionConfig(datapoint.function_name);
  const isReadOnly = useReadOnly();
  const type = function_config?.type || "unknown";

  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(datapoint.updated_at);

  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(type);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Name</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <EditableChip
            label={datapoint.name}
            defaultLabel="—"
            font="mono"
            onSetLabel={isReadOnly ? undefined : onRenameDatapoint}
            tooltipLabel={"Rename"}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Dataset</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Dataset className="text-fg-tertiary" />}
            label={datapoint.dataset_name}
            link={toDatasetUrl(datapoint.dataset_name)}
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
            link={toFunctionUrl(datapoint.function_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {datapoint.source_inference_id && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Inference</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={datapoint.source_inference_id}
              link={toInferenceUrl(datapoint.source_inference_id)}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}

      {datapoint.episode_id && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={datapoint.episode_id}
              link={toEpisodeUrl(datapoint.episode_id)}
              font="mono"
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}

      <BasicInfoItem>
        <BasicInfoItemTitle>Last updated</BasicInfoItemTitle>
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
