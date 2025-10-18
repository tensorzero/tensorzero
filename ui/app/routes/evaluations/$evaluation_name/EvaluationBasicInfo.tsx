import { useFunctionConfig } from "~/context/config";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { toFunctionUrl } from "~/utils/urls";
import type { InferenceEvaluationConfig } from "tensorzero-node";

interface BasicInfoProps {
  evaluation_config: InferenceEvaluationConfig;
}

export default function BasicInfo({ evaluation_config }: BasicInfoProps) {
  const functionName = evaluation_config.function_name;
  const functionConfig = useFunctionConfig(functionName);
  const functionType = functionConfig?.type;
  const functionIconConfig = functionType && getFunctionTypeIcon(functionType);

  return (
    <BasicInfoLayout>
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
    </BasicInfoLayout>
  );
}
