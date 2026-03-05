import { TableItemTime } from "~/components/ui/TableItems";
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
    <div data-testid={testId}>
      <div className="bg-bg-primary border-border overflow-hidden rounded-lg border [&_[data-testid=chat-output]]:!rounded-none [&_[data-testid=chat-output]]:!border-0 [&_[data-testid=chat-output]]:!p-0">
        <div className="bg-bg-secondary text-fg-tertiary flex h-10 items-center border-b px-3 text-sm font-medium">
          {label}
        </div>
        {children}
        {(allTags.length > 0 || timestamp) && (
          <div className="bg-bg-primary text-fg-tertiary flex items-center justify-between border-t px-3 py-2 text-xs">
            <span>
              {allTags.length > 0 ? <TagsPopover tags={allTags} /> : null}
            </span>
            {timestamp && (
              <span>
                Updated <TableItemTime timestamp={timestamp} />
              </span>
            )}
          </div>
        )}
      </div>
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

function DemonstrationPreview({ value }: { value: string }) {
  if (!value) {
    return <NoData />;
  }

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
