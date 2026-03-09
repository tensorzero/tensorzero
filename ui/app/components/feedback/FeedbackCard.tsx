import { TableItemTime } from "~/components/ui/TableItems";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { HoverPopover } from "~/components/ui/hover-popover";
import {
  parseInferenceOutput,
  isJsonOutput,
} from "~/utils/clickhouse/inference";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import {
  TagsPopover,
  filterStringTags,
} from "~/components/feedback/TagsPopover";
import type { FeedbackRow } from "~/types/tensorzero";

function NoData() {
  return <p className="text-fg-muted text-sm">No data</p>;
}

interface FeedbackCardProps {
  label: string;
  children: React.ReactNode;
  tags?: Record<string, unknown>;
  timestamp?: string;
  testId: string;
}

function FeedbackCard({
  label,
  children,
  tags,
  timestamp,
  testId,
}: FeedbackCardProps) {
  const allTags = tags ? filterStringTags(tags) : [];

  return (
    <div
      data-testid={testId}
      className="bg-bg-primary border-border overflow-hidden rounded-md border [&_[data-testid=chat-output]]:!rounded-none [&_[data-testid=chat-output]]:!border-0 [&_[data-testid=chat-output]]:!p-0"
    >
      <div className="bg-bg-secondary text-fg-tertiary flex h-12 items-center border-b px-3 text-sm font-medium">
        {label}
      </div>
      {children}
      {(allTags.length > 0 || timestamp) && (
        <div className="bg-bg-primary text-fg-tertiary flex items-center justify-between border-t px-3 py-2 text-xs">
          {allTags.length > 0 ? <TagsPopover tags={allTags} /> : <span />}
          {timestamp && <CardTimestamp timestamp={timestamp} />}
        </div>
      )}
    </div>
  );
}

interface CommentCardProps {
  comment: Extract<FeedbackRow, { type: "comment" }> | undefined;
}

export function CommentCard({ comment }: CommentCardProps) {
  return (
    <FeedbackCard
      label="Comment"
      tags={comment?.tags}
      timestamp={comment?.timestamp}
      testId={comment ? `feedback-row-${comment.id}` : "feedback-comment"}
    >
      <div className="p-4">
        {comment?.value ? (
          <p className="text-fg-primary text-sm">{comment.value}</p>
        ) : (
          <NoData />
        )}
      </div>
    </FeedbackCard>
  );
}

interface DemonstrationCardProps {
  demonstration: Extract<FeedbackRow, { type: "demonstration" }> | undefined;
}

export function DemonstrationCard({ demonstration }: DemonstrationCardProps) {
  return (
    <FeedbackCard
      label="Demonstration"
      tags={demonstration?.tags}
      timestamp={demonstration?.timestamp}
      testId={
        demonstration
          ? `feedback-row-${demonstration.id}`
          : "feedback-demonstration"
      }
    >
      <div className="p-4">
        {demonstration ? (
          <DemonstrationPreview value={demonstration.value} />
        ) : (
          <NoData />
        )}
      </div>
    </FeedbackCard>
  );
}

function CardTimestamp({ timestamp }: { timestamp: string }) {
  return (
    <HoverPopover
      side="top"
      align="end"
      className="text-xs"
      trigger={
        <span className="cursor-default">
          <TableItemTime timestamp={timestamp} />
        </span>
      }
    >
      <div className="flex flex-col gap-1">
        <div className="text-fg-tertiary">Last updated</div>
        <TimestampTooltip timestamp={timestamp} />
      </div>
    </HoverPopover>
  );
}

function DemonstrationPreview({ value }: { value: string }) {
  try {
    const parsedOutput = parseInferenceOutput(value);
    return isJsonOutput(parsedOutput) ? (
      <JsonOutputElement output={parsedOutput} />
    ) : (
      <ChatOutputElement output={parsedOutput} />
    );
  } catch {
    return (
      <pre className="text-fg-primary line-clamp-3 whitespace-pre-wrap font-mono text-sm">
        {value}
      </pre>
    );
  }
}
