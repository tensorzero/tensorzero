import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { DotSeparator } from "~/components/ui/DotSeparator";
import { TableItemTime } from "~/components/ui/TableItems";
import { ToolEventId } from "~/components/autopilot/EventStream";
import { cn } from "~/utils/common";
import type { EventPayloadUserQuestions } from "~/types/tensorzero";

export function SkippedQuestionCard({
  payload,
  eventId,
  timestamp,
  className,
}: {
  payload: EventPayloadUserQuestions;
  eventId: string;
  timestamp: string;
  className?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "border-border bg-bg-secondary flex flex-col rounded-md border",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <button
          type="button"
          onClick={() => setIsExpanded((e) => !e)}
          aria-expanded={isExpanded}
          aria-label={
            isExpanded ? "Collapse question details" : "Expand question details"
          }
          className="inline-flex cursor-pointer items-center gap-2 text-left"
        >
          <span className="inline-flex items-center gap-2 text-sm font-medium">
            Question
            <DotSeparator />
            Skipped
          </span>
          <span
            className={cn(
              "text-fg-muted inline-flex transition-transform duration-200",
              isExpanded ? "rotate-90" : "rotate-0",
            )}
          >
            <ChevronRight className="h-4 w-4" />
          </span>
        </button>
        <div className="text-fg-muted flex items-center gap-1.5 text-xs">
          <ToolEventId id={eventId} />
          <DotSeparator />
          <TableItemTime timestamp={timestamp} />
        </div>
      </div>

      {isExpanded && (
        <div className="flex flex-col gap-3 px-4 pb-3">
          {payload.questions.map((q) => (
            <div key={q.id} className="flex flex-col gap-0.5">
              <span className="text-fg-muted text-xs font-medium">
                {q.header}
              </span>
              <span className="text-fg-muted text-sm italic">{q.question}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
