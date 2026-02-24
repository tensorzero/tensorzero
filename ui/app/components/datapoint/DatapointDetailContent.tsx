import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { TagsTable } from "~/components/tags/TagsTable";
import {
  SectionsGroup,
  SectionLayout,
  SectionHeader,
} from "~/components/layout/PageLayout";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Dataset, Calendar } from "~/components/icons/Icons";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { toDatasetUrl, toFunctionUrl } from "~/utils/urls";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useFunctionConfig } from "~/context/config";
import type { DatapointDetailData } from "~/routes/api/datapoint/$id/route";

interface DatapointDetailContentProps {
  data: DatapointDetailData;
}

export function DatapointDetailContent({ data }: DatapointDetailContentProps) {
  const { datapoint, resolvedInput, dataset_name, function_name } = data;
  const functionConfig = useFunctionConfig(function_name);
  const functionType = functionConfig?.type || "unknown";
  const functionIconConfig = getFunctionTypeIcon(functionType);

  return (
    <SectionsGroup>
      <SectionLayout>
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Dataset</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                icon={<Dataset className="text-fg-tertiary" />}
                label={dataset_name}
                link={toDatasetUrl(dataset_name)}
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
                label={function_name}
                secondaryLabel={functionType}
                link={toFunctionUrl(function_name)}
                font="mono"
              />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Type</BasicInfoItemTitle>
            <BasicInfoItemContent>
              {datapoint.type === "chat" ? "Chat" : "JSON"}
            </BasicInfoItemContent>
          </BasicInfoItem>

          {datapoint.is_custom && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Custom</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="cursor-help">
                      Custom
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    This datapoint is not based on a historical inference. It
                    was either edited or created manually.
                  </TooltipContent>
                </Tooltip>
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {datapoint.staled_at && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Status</BasicInfoItemTitle>
              <BasicInfoItemContent>
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
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          <BasicInfoItem>
            <BasicInfoItemTitle>Last updated</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                icon={<Calendar className="text-fg-tertiary" />}
                label={formatDateWithSeconds(new Date(datapoint.updated_at))}
                tooltip={<TimestampTooltip timestamp={datapoint.updated_at} />}
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Input" />
        <InputElement input={resolvedInput} />
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Output" />
        {datapoint.type === "chat" ? (
          <ChatOutputElement output={datapoint.output} />
        ) : (
          <JsonOutputElement
            output={datapoint.output}
            outputSchema={datapoint.output_schema}
          />
        )}
      </SectionLayout>

      {datapoint.tags && Object.keys(datapoint.tags).length > 0 && (
        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsTable tags={datapoint.tags} isEditing={false} />
        </SectionLayout>
      )}
    </SectionsGroup>
  );
}
