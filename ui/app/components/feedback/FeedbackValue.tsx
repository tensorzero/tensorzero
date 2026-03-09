import { useState, type ReactNode } from "react";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { MetricConfig, FeedbackRow } from "~/types/tensorzero";
import { getFeedbackIcon } from "~/utils/icon";
import { UserFeedback } from "../icons/Icons";
import {
  PageLayout,
  PageHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { TextMessage } from "~/components/layout/SnippetContent";
import {
  parseInferenceOutput,
  isJsonOutput,
} from "~/utils/clickhouse/inference";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";

interface ValueItemProps {
  iconType:
    | "success"
    | "failure"
    | "default"
    | "unknown"
    | "float"
    | "comment"
    | "demonstration";
  children: ReactNode;
  onClick?: (event: React.MouseEvent) => void;
}

function ValueItem({ iconType, children, onClick }: ValueItemProps) {
  const { icon, iconBg } = getFeedbackIcon(iconType);

  return (
    <div
      className={
        onClick
          ? "flex cursor-pointer items-center gap-2 transition-colors duration-300 hover:text-gray-500"
          : "flex items-center gap-2"
      }
      onClick={onClick}
    >
      <div
        className={`flex h-5 w-5 min-w-[1.25rem] items-center justify-center rounded-md ${iconBg}`}
      >
        {icon}
      </div>
      {children}
    </div>
  );
}

function ValueItemText({ children }: { children: ReactNode }) {
  return (
    <span className="overflow-hidden text-ellipsis whitespace-nowrap">
      {children}
    </span>
  );
}

function CommentModal({ feedback }: { feedback: FeedbackRow }) {
  if (feedback.type !== "comment" || typeof feedback.value !== "string") {
    return <div>Invalid comment feedback</div>;
  }

  return (
    <PageLayout>
      <PageHeader eyebrow="Comment" name={feedback.id} />
      <SectionsGroup>
        <SectionLayout>
          <SnippetLayout>
            <SnippetContent maxHeight="Content">
              <SnippetMessage>
                <TextMessage content={feedback.value} />
              </SnippetMessage>
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

function DemonstrationModal({ feedback }: { feedback: FeedbackRow }) {
  if (feedback.type !== "demonstration" || typeof feedback.value !== "string") {
    return <div>Invalid demonstration feedback</div>;
  }

  const parsedOutput = parseInferenceOutput(feedback.value);

  return (
    <PageLayout>
      <PageHeader eyebrow="Demonstration" name={feedback.id} />
      <SectionsGroup>
        <SectionLayout>
          {isJsonOutput(parsedOutput) ? (
            <JsonOutputElement output={parsedOutput} />
          ) : (
            <ChatOutputElement output={parsedOutput} />
          )}
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

interface FeedbackValueProps {
  feedback: FeedbackRow;
  metric?: MetricConfig;
}

export default function FeedbackValue({
  feedback,
  metric,
}: FeedbackValueProps) {
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const handleClick = (event: React.MouseEvent) => {
    if (feedback.type === "comment" || feedback.type === "demonstration") {
      event.stopPropagation();
      setIsSheetOpen(true);
    }
  };

  const isHumanFeedback =
    feedback.tags["tensorzero::human_feedback"] === "true";

  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : "unknown";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    const failure =
      (feedback.value === true && optimize === "min") ||
      (feedback.value === false && optimize === "max");

    let status: "success" | "failure" | "default" = "default";

    if (success) {
      status = "success";
    } else if (failure) {
      status = "failure";
    }

    return (
      <ValueItem iconType={status === "default" ? "unknown" : status}>
        <ValueItemText>{feedback.value ? "True" : "False"}</ValueItemText>
        {isHumanFeedback && <UserFeedback />}
      </ValueItem>
    );
  }

  if (feedback.type === "float" && typeof feedback.value === "number") {
    return (
      <ValueItem iconType="float">
        <ValueItemText>{feedback.value.toFixed(3)}</ValueItemText>
        {isHumanFeedback && <UserFeedback />}
      </ValueItem>
    );
  }

  if (feedback.type === "comment" && typeof feedback.value === "string") {
    return (
      <>
        <ValueItem iconType="comment" onClick={handleClick}>
          <ValueItemText>{feedback.value}</ValueItemText>
          {isHumanFeedback && <UserFeedback />}
        </ValueItem>
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetContent className="bg-bg-secondary overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
            <CommentModal feedback={feedback} />
          </SheetContent>
        </Sheet>
      </>
    );
  }

  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    return (
      <>
        <ValueItem iconType="demonstration" onClick={handleClick}>
          <ValueItemText>
            <span className="font-mono">{feedback.value}</span>
          </ValueItemText>
          {isHumanFeedback && <UserFeedback />}
        </ValueItem>
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetContent className="bg-bg-secondary w-full overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
            <DemonstrationModal feedback={feedback} />
          </SheetContent>
        </Sheet>
      </>
    );
  }

  return <div className="text-red-500">Invalid feedback type</div>;
}
