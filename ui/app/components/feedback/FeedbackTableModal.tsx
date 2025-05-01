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
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { TextMessage } from "~/components/layout/SnippetContent";
import Output from "~/components/inference/NewOutput";

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

export function DemonstrationModal({ feedback }: FeedbackTableModalProps) {
  if (feedback.type !== "demonstration" || typeof feedback.value !== "string") {
    return <div>Invalid demonstration feedback</div>;
  }

  // Try to parse the demonstration value as JSON for the parsed property
  let parsedValue = null;
  try {
    parsedValue = JSON.parse(feedback.value);
  } catch {
    // If parsing fails, keep parsedValue as null
  }

  return (
    <PageLayout>
      <PageHeader label="Demonstration" name={feedback.id} />

      <SectionsGroup>
        <SectionLayout>
          <Output
            output={{
              raw: feedback.value,
              parsed: parsedValue,
            }}
          />
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
