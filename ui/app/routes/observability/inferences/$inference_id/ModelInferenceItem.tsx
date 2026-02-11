import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { InputElement } from "~/components/input_output/InputElement";
import type {
  Detail,
  Input,
  InputMessageContent,
  StoragePath,
} from "~/types/tensorzero";
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
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import ModelInferenceOutput from "~/components/input_output/ModelInferenceOutput";

interface ModelInferenceItemProps {
  inference: ParsedModelInferenceRow;
}

export function ModelInferenceItem({ inference }: ModelInferenceItemProps) {
  return (
    <PageLayout>
      <PageHeader eyebrow="Model Inference" name={inference.id}>
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
                  label={`${inference.input_tokens === undefined ? "null" : inference.input_tokens} tok`}
                  tooltip="Input Tokens"
                />
                <Chip
                  icon={<Output className="text-fg-tertiary" />}
                  label={`${inference.output_tokens === undefined ? "null" : inference.output_tokens} tok`}
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
                tooltip={<TimestampTooltip timestamp={inference.timestamp} />}
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputElement input={modelInferenceToInput(inference)} />
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

/**
 * Converts a ParsedModelInferenceRow's input fields into the modern Input type
 * expected by InputElement. The display types (from the ClickHouse/resolve layer)
 * are structurally compatible for text/tool_call/tool_result/thought, but files
 * need reshaping from the resolved base64 format to the File union type.
 */
function modelInferenceToInput(inference: ParsedModelInferenceRow): Input {
  return {
    system: inference.system ?? undefined,
    messages: inference.input_messages.map((msg) => ({
      role: msg.role,
      content: msg.content.map(
        (block): InputMessageContent =>
          displayContentToInputContent(
            block as { type: string } & Record<string, unknown>,
          ),
      ),
    })),
  };
}

function displayContentToInputContent(
  block: { type: string } & Record<string, unknown>,
): InputMessageContent {
  switch (block.type) {
    case "text":
    case "tool_call":
    case "tool_result":
    case "thought":
    case "unknown":
      return block as InputMessageContent;
    case "file": {
      const file = block.file as { data: string; mime_type: string };
      return {
        type: "file",
        file_type: "object_storage",
        data: file.data,
        mime_type: file.mime_type,
        storage_path: block.storage_path as StoragePath,
        source_url: (block.source_url as string) ?? undefined,
        detail: (block.detail as Detail) ?? undefined,
        filename: (block.filename as string) ?? undefined,
      };
    }
    case "file_error": {
      const file = block.file as { mime_type: string };
      return {
        type: "file",
        file_type: "object_storage_error",
        error: (block.error as string) ?? undefined,
        mime_type: file.mime_type,
        storage_path: block.storage_path as StoragePath,
        source_url: (block.source_url as string) ?? undefined,
        detail: (block.detail as Detail) ?? undefined,
        filename: (block.filename as string) ?? undefined,
      };
    }
    default:
      return {
        type: "unknown",
        data: JSON.parse(JSON.stringify(block)),
      };
  }
}
