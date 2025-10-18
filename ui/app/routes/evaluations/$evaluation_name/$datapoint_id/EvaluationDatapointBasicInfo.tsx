import { useFunctionConfig } from "~/context/config";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { InferenceEvaluationConfig } from "tensorzero-node";
import EditableChip from "~/components/ui/EditableChip";
import {
  toEvaluationUrl,
  toFunctionUrl,
  toDatasetUrl,
  toDatapointUrl,
} from "~/utils/urls";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface BasicInfoProps {
  evaluation_name: string;
  evaluation_config: InferenceEvaluationConfig;
  dataset_name: string;
  datapoint_id: string;
  datapoint_name: string | null;
  datapoint_staled_at: string | null;
  onRenameDatapoint?: (newName: string) => void | Promise<void>;
}

export default function BasicInfo({
  evaluation_name,
  evaluation_config,
  dataset_name,
  datapoint_id,
  datapoint_name,
  datapoint_staled_at,
  onRenameDatapoint,
}: BasicInfoProps) {
  const functionName = evaluation_config.function_name;
  const functionConfig = useFunctionConfig(functionName);
  const functionType = functionConfig?.type;
  const functionIconConfig = functionType
    ? getFunctionTypeIcon(functionType)
    : null;

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Name</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <EditableChip
            label={datapoint_name}
            defaultLabel="—"
            font="mono"
            onSetLabel={onRenameDatapoint}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
      <BasicInfoItem>
        <BasicInfoItemTitle>Evaluation</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={evaluation_name}
            link={toEvaluationUrl(evaluation_name)}
            font="mono"
          />
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
              link={toFunctionUrl(functionName)}
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
              <TooltipProvider delayDuration={200}>
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
              </TooltipProvider>
            )}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
