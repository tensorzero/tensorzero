import type { FunctionConfig } from "tensorzero-node";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";

interface BasicInfoProps {
  functionConfig: FunctionConfig;
}

export default function BasicInfo({ functionConfig }: BasicInfoProps) {
  return (
    <BasicInfoLayout>
      {functionConfig.description && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Description</BasicInfoItemTitle>
          <BasicInfoItemContent>
            {functionConfig.description}
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}
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
              <Chip
                label={
                  typeof functionConfig.tool_choice === "object"
                    ? functionConfig.tool_choice.specific
                    : functionConfig.tool_choice
                }
                font="mono"
              />
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
