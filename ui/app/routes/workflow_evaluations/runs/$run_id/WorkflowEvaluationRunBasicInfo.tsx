import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Calendar } from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import type { WorkflowEvaluationRun } from "~/utils/clickhouse/workflow_evaluations";
import KVChip from "~/components/ui/KVChip";
import { CommitHash } from "~/components/ui/CommitHash";
import {
  toFunctionUrl,
  toVariantUrl,
  toWorkflowEvaluationProjectUrl,
} from "~/utils/urls";

// Create timestamp tooltip component
const createTimestampTooltip = (timestamp: string | number | Date) => {
  const { formattedDate, formattedTime, relativeTime } =
    getTimestampTooltipData(timestamp);

  return (
    <div className="flex flex-col gap-1">
      <div>{formattedDate}</div>
      <div>{formattedTime}</div>
      <div>{relativeTime}</div>
    </div>
  );
};

interface BasicInfoProps {
  workflowEvaluationRun: WorkflowEvaluationRun;
  count: number;
}

export default function BasicInfo({
  workflowEvaluationRun,
  count,
}: BasicInfoProps) {
  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(
    workflowEvaluationRun.timestamp,
  );

  const filteredTags = Object.entries(workflowEvaluationRun.tags).filter(
    ([k]) => !k.startsWith("tensorzero::"),
  );

  const commitHash = workflowEvaluationRun.tags["tensorzero::git_commit_hash"];

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        {workflowEvaluationRun.name ? (
          <>
            <BasicInfoItemTitle>Name</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={workflowEvaluationRun.name} font="mono" />
            </BasicInfoItemContent>
          </>
        ) : null}
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>ID</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={workflowEvaluationRun.id} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {workflowEvaluationRun.project_name ? (
        <BasicInfoItem>
          <BasicInfoItemTitle>Project</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={workflowEvaluationRun.project_name}
              font="mono"
              link={toWorkflowEvaluationProjectUrl(
                workflowEvaluationRun.project_name,
              )}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      ) : null}

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant Pins</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex flex-wrap gap-1">
            {Object.entries(workflowEvaluationRun.variant_pins).map(
              ([k, v]) => (
                <KVChip
                  key={k}
                  k={k}
                  v={v}
                  k_href={toFunctionUrl(k)}
                  v_href={toVariantUrl(k, v)}
                />
              ),
            )}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      {commitHash && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Git Commit</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <CommitHash tags={workflowEvaluationRun.tags} />
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}

      {filteredTags.length > 0 && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Tags</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <div className="flex flex-wrap gap-1">
              {filteredTags.map(([k, v]) => (
                <KVChip key={k} k={k} v={v} />
              ))}
            </div>
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}

      <BasicInfoItem>
        <BasicInfoItemTitle>Episode Count</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={`${count}`} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(
              new Date(workflowEvaluationRun.timestamp),
            )}
            tooltip={timestampTooltip}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
