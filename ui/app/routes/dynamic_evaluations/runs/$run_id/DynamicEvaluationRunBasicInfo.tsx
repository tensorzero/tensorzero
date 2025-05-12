import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { Calendar } from "~/components/icons/Icons";
import { formatDateWithSeconds, getTimestampTooltipData } from "~/utils/date";
import type { DynamicEvaluationRun } from "~/utils/clickhouse/dynamic_evaluations";
import KVChip from "~/components/ui/KVChip";
import { CommitHash } from "~/components/ui/CommitHash";

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
  dynamicEvaluationRun: DynamicEvaluationRun;
  count: number;
}

export default function BasicInfo({
  dynamicEvaluationRun,
  count,
}: BasicInfoProps) {
  // Create timestamp tooltip
  const timestampTooltip = createTimestampTooltip(
    dynamicEvaluationRun.timestamp,
  );

  const filteredTags = Object.entries(dynamicEvaluationRun.tags).filter(
    ([k]) => !k.startsWith("tensorzero::"),
  );

  const commitHash = dynamicEvaluationRun.tags["tensorzero::git_commit_hash"];

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        {dynamicEvaluationRun.name ? (
          <>
            <BasicInfoItemTitle>Name</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={dynamicEvaluationRun.name} font="mono" />
            </BasicInfoItemContent>
          </>
        ) : null}
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>ID</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={dynamicEvaluationRun.id} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {dynamicEvaluationRun.project_name ? (
        <BasicInfoItem>
          <BasicInfoItemTitle>Project</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip
              label={dynamicEvaluationRun.project_name}
              font="mono"
              link={`/dynamic_evaluations/projects/${dynamicEvaluationRun.project_name}`}
            />
          </BasicInfoItemContent>
        </BasicInfoItem>
      ) : null}

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant Pins</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <div className="flex flex-wrap gap-1">
            {Object.entries(dynamicEvaluationRun.variant_pins).map(([k, v]) => (
              <KVChip
                key={k}
                k={k}
                v={v}
                k_href={`/observability/functions/${k}`}
                v_href={`/observability/functions/${k}/variants/${v}`}
              />
            ))}
          </div>
        </BasicInfoItemContent>
      </BasicInfoItem>

      {commitHash && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Git Commit</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <CommitHash tags={dynamicEvaluationRun.tags} />
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
              new Date(dynamicEvaluationRun.timestamp),
            )}
            tooltip={timestampTooltip}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
