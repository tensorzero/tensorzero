import { useFunctionConfig } from "~/context/config";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { StaticEvaluationConfig } from "tensorzero-node";

interface BasicInfoProps {
  evaluation_name: string;
  evaluation_config: StaticEvaluationConfig;
  dataset_name: string;
  datapoint_name: string | null;
}

export default function BasicInfo({
  evaluation_name,
  evaluation_config,
  dataset_name,
  datapoint_name,
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
          {/* TODO: support editing names */}
          <Chip
            label={datapoint_name || "-"}
            font="mono"
            />
        </BasicInfoItemContent>
      </BasicInfoItem>
      <BasicInfoItem>
        <BasicInfoItemTitle>Evaluation</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={evaluation_name}
            link={`/evaluations/${evaluation_name}`}
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
              link={`/observability/functions/${functionName}`}
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
            link={`/datasets/${dataset_name}`}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
