import { useConfig } from "~/context/config";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { EvaluationConfig } from "~/utils/config/evaluations";

interface BasicInfoProps {
  evaluation_config: EvaluationConfig;
}

export default function BasicInfo({ evaluation_config }: BasicInfoProps) {
  const config = useConfig();

  const functionName = evaluation_config.function_name;
  const functionConfig = config.functions[functionName];
  const functionType = functionConfig?.type;
  const functionIconConfig = getFunctionTypeIcon(functionType);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={functionName}
            secondaryLabel={`· ${functionType}`}
            link={`/observability/functions/${functionName}`}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
