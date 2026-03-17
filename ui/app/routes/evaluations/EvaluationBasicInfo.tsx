import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { toFunctionUrl } from "~/utils/urls";

interface BasicInfoProps {
  functionName: string;
  functionType: "chat" | "json";
  snapshotHash?: string;
}

export default function BasicInfo({
  functionName,
  functionType,
  snapshotHash,
}: BasicInfoProps) {
  const functionIconConfig = getFunctionTypeIcon(functionType);

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
              link={toFunctionUrl(functionName, snapshotHash)}
              font="mono"
            />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
