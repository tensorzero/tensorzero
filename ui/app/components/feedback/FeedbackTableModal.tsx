import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import {
  PageLayout,
  PageHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeMessage, TextMessage } from "~/components/layout/SnippetContent";

interface FeedbackTableModalProps {
  feedback: FeedbackRow;
}

export function CommentModal({ feedback }: FeedbackTableModalProps) {
  if (feedback.type !== "comment" || typeof feedback.value !== "string") {
    return <div>Invalid comment feedback</div>;
  }

  return (
    <PageLayout>
      <PageHeader label="Comment" name={feedback.id} />

      <SectionsGroup>
        <SectionLayout>
          <SnippetLayout>
            <SnippetContent maxHeight="Content">
              <TextMessage content={feedback.value} />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

export function DemonstrationModal({ feedback }: FeedbackTableModalProps) {
  if (feedback.type !== "demonstration" || typeof feedback.value !== "string") {
    return <div>Invalid demonstration feedback</div>;
  }

  return (
    <PageLayout>
      <PageHeader label="Demonstration" name={feedback.id} />

      <SectionsGroup>
        <SectionLayout>
          <SnippetLayout>
            <SnippetContent maxHeight="Content">
              <CodeMessage showLineNumbers content={feedback.value} />
            </SnippetContent>
          </SnippetLayout>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

export default function FeedbackTableModal({
  feedback,
}: FeedbackTableModalProps) {
  if (feedback.type === "comment") {
    return <CommentModal feedback={feedback} />;
  }

  if (feedback.type === "demonstration") {
    return <DemonstrationModal feedback={feedback} />;
  }

  return <div>No feedback data</div>;
}
