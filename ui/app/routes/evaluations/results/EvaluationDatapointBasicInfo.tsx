import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import EditableChip from "~/components/ui/EditableChip";
import { toFunctionUrl, toDatasetUrl, toDatapointUrl } from "~/utils/urls";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { useReadOnly } from "~/context/read-only";

interface BasicInfoProps {
  evaluation_name: string;
  functionName: string;
  functionType: "chat" | "json";
  dataset_name: string;
  datapoint_id: string;
  datapoint_name?: string;
  datapoint_staled_at?: string;
  onRenameDatapoint?: (newName: string) => void | Promise<void>;
  snapshotHash?: string;
}

export default function BasicInfo({
  evaluation_name,
  functionName,
  functionType,
  dataset_name,
  datapoint_id,
  datapoint_name,
  datapoint_staled_at,
  onRenameDatapoint,
  snapshotHash,
}: BasicInfoProps) {
  const isReadOnly = useReadOnly();
  const functionIconConfig = getFunctionTypeIcon(functionType);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Name</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <EditableChip
            label={datapoint_name}
            defaultLabel="—"
            font="mono"
            onSetLabel={isReadOnly ? undefined : onRenameDatapoint}
            tooltipLabel={"Rename"}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
      <BasicInfoItem>
        <BasicInfoItemTitle>Evaluation</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={evaluation_name} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          {functionIconConfig && (
            <Chip
              icon={functionIconConfig.icon}
              iconBg={functionIconConfig.iconBg}
              label={functionName}
              secondaryLabel={`· ${functionType}`}
              link={toFunctionUrl(functionName, snapshotHash)}
              font="mono"
            />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Dataset</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={dataset_name}
            link={toDatasetUrl(dataset_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Datapoint</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex items-center gap-2">
            <Chip
              label={datapoint_id}
              link={toDatapointUrl(dataset_name, datapoint_id)}
              font="mono"
            />
            {datapoint_staled_at && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="cursor-help">
                    Stale
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  This datapoint has since been edited or deleted.
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
