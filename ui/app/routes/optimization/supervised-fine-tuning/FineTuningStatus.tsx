/**
 * Component that displays the status and details of a Language Model Fine-Tuning job.
 * Shows metadata like job ID, status, base model, function, metrics, and progress.
 * Includes links to external job details and raw data visualization.
 */

import { ExternalLink } from "lucide-react";
import { Calendar, Function } from "~/components/icons/Icons";
import type { SFTJobStatus } from "~/utils/supervised_fine_tuning/common";
import { extractTimestampFromUUIDv7 } from "~/utils/common";
import { RawDataAccordion } from "./RawDataAccordion";
import { ProgressIndicator } from "./ProgressIndicator";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import {
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { SFTResult } from "./SFTResult";

export default function LLMFineTuningStatus({
  status,
  result,
}: {
  status: SFTJobStatus;
  result: string | null;
}) {
  if (status.status === "idle") return null;
  const createdAt = extractTimestampFromUUIDv7(status.formData.jobId);
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Results" />
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Base Model</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={status.formData.model.name} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Function</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                link={[
                  "/observability/functions/:function_name",
                  { function_name: status.formData.function },
                ]}
                icon={<Function className="text-fg-tertiary" />}
                label={status.formData.function}
                font="mono"
              />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Metric</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={status.formData.metric ?? "None"} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Prompt</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={status.formData.variant} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Created</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                icon={<Calendar className="text-fg-tertiary" />}
                label={createdAt.toLocaleString()}
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </SectionLayout>

      {/* hide if not available from provider, eg fireworks */}
      {status.status === "running" &&
        "estimatedCompletionTime" in status &&
        status.estimatedCompletionTime && (
          <SectionLayout>
            <SectionHeader heading="Estimated completion" />
            <div className="max-w-lg space-y-2">
              <ProgressIndicator
                createdAt={createdAt}
                estimatedCompletion={
                  new Date(status.estimatedCompletionTime * 1000)
                }
              />
            </div>
          </SectionLayout>
        )}
      <SFTResult finalResult={result} />

      <SectionLayout>
        <a
          href={status.jobUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary inline-flex items-center text-sm hover:underline"
        >
          <SectionHeader heading="Job details"></SectionHeader>
          <ExternalLink className="ml-2 h-5 w-5" />
        </a>

        <RawDataAccordion rawData={status.rawData} />
      </SectionLayout>
    </SectionsGroup>
  );
}
