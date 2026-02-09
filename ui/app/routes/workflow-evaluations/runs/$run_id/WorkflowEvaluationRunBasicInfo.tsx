import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Calendar } from "~/components/icons/Icons";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import type { WorkflowEvaluationRunWithEpisodeCount } from "~/types/tensorzero";
import KVChip from "~/components/ui/KVChip";
import { CommitHash } from "~/components/ui/CommitHash";
import {
  toFunctionUrl,
  toVariantUrl,
  toWorkflowEvaluationProjectUrl,
} from "~/utils/urls";

interface BasicInfoProps {
  workflowEvaluationRun: WorkflowEvaluationRunWithEpisodeCount;
  count: number;
}

export default function BasicInfo({
  workflowEvaluationRun,
  count,
}: BasicInfoProps) {
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
            {Object.entries(workflowEvaluationRun.variant_pins)
              .filter((entry): entry is [string, string] => entry[1] != null)
              .map(([k, v]) => (
                <KVChip
                  key={k}
                  k={k}
                  v={v}
                  k_href={toFunctionUrl(k)}
                  v_href={toVariantUrl(k, v)}
                />
              ))}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      {commitHash && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Git Commit</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <CommitHash
              tags={
                workflowEvaluationRun.tags as unknown as Record<string, string>
              }
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}

      {filteredTags.length > 0 && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Tags</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <div className="flex flex-wrap gap-1">
              {filteredTags
                .filter((entry): entry is [string, string] => entry[1] != null)
                .map(([k, v]) => (
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
            tooltip={
              <TimestampTooltip timestamp={workflowEvaluationRun.timestamp} />
            }
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
