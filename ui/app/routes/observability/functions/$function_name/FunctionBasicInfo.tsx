import type { FunctionConfig } from "~/utils/config/function";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";

interface BasicInfoProps {
  functionConfig: FunctionConfig;
}

export default function BasicInfo({ functionConfig }: BasicInfoProps) {
  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(functionConfig.type);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Type</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={functionConfig.type}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {/* Fields specific to chat type */}
      {functionConfig.type === "chat" && (
        <>
          <BasicInfoItem>
            <BasicInfoItemTitle>Tools</BasicInfoItemTitle>
            <BasicInfoItemContent>
              {functionConfig.tools?.length ? (
                <div className="flex flex-wrap gap-1">
                  {functionConfig.tools.map((tool) => (
                    <Chip key={tool} label={tool} font="mono" />
                  ))}
                </div>
              ) : (
                <Chip label="none" font="mono" />
              )}
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Tool Choice</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={functionConfig.tool_choice} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Parallel Tool Calls</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                label={functionConfig.parallel_tool_calls ? "true" : "false"}
                font="mono"
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </>
      )}
    </BasicInfoLayout>
  );
}
