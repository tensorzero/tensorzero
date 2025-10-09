/**
 * Component that displays the status and details of a Language Model Fine-Tuning job.
 * Shows metadata like job ID, status, base model, function, metrics, and progress.
 * Includes links to external job details and raw data visualization.
 */

import { ExternalLink } from "lucide-react";
import { Calendar, Function } from "~/components/icons/Icons";
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
import type { SFTFormValues } from "./types";
import type {
  OptimizationJobHandle,
  OptimizationJobInfo,
} from "tensorzero-node";
import { toFunctionUrl } from "~/utils/urls";

export default function LLMFineTuningStatus({
  status,
  formData,
  result,
  jobHandle,
}: {
  status: OptimizationJobInfo;
  formData: SFTFormValues;
  result: string | null;
  jobHandle: OptimizationJobHandle;
}) {
  const createdAt = extractTimestampFromUUIDv7(formData.jobId);
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Results" />
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Base Model</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={formData.model.name} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Function</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                link={toFunctionUrl(formData.function)}
                icon={<Function className="text-fg-tertiary" />}
                label={formData.function}
                font="mono"
              />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Metric</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={formData.metric ?? "None"} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Prompt</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={formData.variant} font="mono" />
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
      {status.status === "pending" && status.estimated_finish && (
        <SectionLayout>
          <SectionHeader heading="Estimated completion" />
          <div className="max-w-lg space-y-2">
            <ProgressIndicator
              createdAt={createdAt}
              estimatedCompletion={status.estimated_finish}
            />
          </div>
        </SectionLayout>
      )}
      <SFTResult finalResult={result} />

      <SectionLayout>
        {"job_url" in jobHandle && (
          <a
            href={jobHandle.job_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary inline-flex items-center text-sm hover:underline"
          >
            <SectionHeader heading="Job details"></SectionHeader>
            <ExternalLink className="ml-2 h-5 w-5" />
          </a>
        )}

        <RawDataAccordion rawData={status} />
      </SectionLayout>
    </SectionsGroup>
  );
}
