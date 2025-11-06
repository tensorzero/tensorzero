import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import Input from "~/components/inference/Input";
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
import {
  Timer,
  InputIcon,
  Output,
  Calendar,
  Cached,
} from "~/components/icons/Icons";
import Chip from "~/components/ui/Chip";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import ModelInferenceOutput from "~/components/inference/ModelInferenceOutput";

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
                  icon={<InputIcon className="text-fg-tertiary" />}
                  label={`${inference.input_tokens} tok`}
                  tooltip="Input Tokens"
                />
                <Chip
                  icon={<Output className="text-fg-tertiary" />}
                  label={`${inference.output_tokens} tok`}
                  tooltip="Output Tokens"
                />
                {inference.response_time_ms !== null && (
                  <Chip
                    icon={<Timer className="text-fg-tertiary" />}
                    label={`${inference.response_time_ms} ms`}
                    tooltip="Response Time"
                  />
                )}
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
          <Input
            system={inference.system}
            messages={inference.input_messages}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <ModelInferenceOutput output={inference.output} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Raw Request" />
          <SnippetLayout>
            <SnippetContent maxHeight={400}>
              <CodeEditor
                allowedLanguages={["json"]}
                value={(() => {
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
                readOnly
              />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Raw Response" />
          <SnippetLayout>
            <SnippetContent maxHeight={400}>
              <CodeEditor
                allowedLanguages={["json"]}
                value={(() => {
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
                readOnly
              />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
