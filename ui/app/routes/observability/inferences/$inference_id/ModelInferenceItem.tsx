import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import ModelInput from "~/components/model/ModelInput";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import {
  PageLayout,
  PageHeader,
  SectionLayout,
  SectionHeader,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { Timer, Input, Output, Calendar, Cached } from "~/components/icons/Icons";
import Chip from "~/components/ui/Chip";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeMessage } from "~/components/layout/SnippetContent";

interface ModelInferenceItemProps {
  inference: ParsedModelInferenceRow;
}

export function ModelInferenceItem({ inference }: ModelInferenceItemProps) {
  // Create timestamp tooltip
  const { formattedDate, formattedTime, relativeTime } =
    getTimestampTooltipData(inference.timestamp);
  const timestampTooltip = (
    <div className="flex flex-col gap-1">
      <div>{formattedDate}</div>
      <div>{formattedTime}</div>
      <div>{relativeTime}</div>
    </div>
  );

  return (
    <PageLayout>
      <PageHeader label="Model Inference" name={inference.id}>
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Model</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={inference.model_name} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Model Provider</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={inference.model_provider_name} />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <div className="flex flex-row gap-1">
                <Chip
                  icon={<Input className="text-fg-tertiary" />}
                  label={`${inference.input_tokens} tok`}
                  tooltip="Input Tokens"
                />
                <Chip
                  icon={<Output className="text-fg-tertiary" />}
                  label={`${inference.output_tokens} tok`}
                  tooltip="Output Tokens"
                />
                <Chip
                  icon={<Timer className="text-fg-tertiary" />}
                  label={`${inference.response_time_ms} ms`}
                  tooltip="Response Time"
                />
                {inference.cached && (
                  <Chip
                    icon={<Cached className="text-fg-tertiary" />}
                    label="Cached"
                    tooltip="Model Inference was cached by TensorZero"
                  />
                )}
              </div>
            </BasicInfoItemContent>
          </BasicInfoItem>

          {inference.ttft_ms && (
            <BasicInfoItem>
              <BasicInfoItemTitle>TTFT</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip
                  icon={<Timer className="text-fg-tertiary" />}
                  label={`${inference.ttft_ms} ms`}
                  tooltip="Time To First Token"
                />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          <BasicInfoItem>
            <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                icon={<Calendar className="text-fg-tertiary" />}
                label={formatDateWithSeconds(new Date(inference.timestamp))}
                tooltip={timestampTooltip}
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <ModelInput
            input_messages={inference.input_messages}
            system={inference.system}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <SnippetLayout>
            <SnippetContent maxHeight={400}>
              <CodeMessage
                showLineNumbers
                content={JSON.stringify(inference.output, null, 2)}
              />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Raw Request" />
          <SnippetLayout>
            <SnippetContent maxHeight={400}>
              <CodeMessage
                showLineNumbers
                content={(() => {
                  try {
                    return JSON.stringify(
                      JSON.parse(inference.raw_request),
                      null,
                      2,
                    );
                  } catch {
                    return inference.raw_request;
                  }
                })()}
              />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Raw Response" />
          <SnippetLayout>
            <SnippetContent maxHeight={400}>
              <CodeMessage
                showLineNumbers
                content={(() => {
                  try {
                    return JSON.stringify(
                      JSON.parse(inference.raw_response),
                      null,
                      2,
                    );
                  } catch {
                    return inference.raw_response;
                  }
                })()}
              />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
