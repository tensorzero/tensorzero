import type { FeedbackDetailData } from "~/routes/api/feedback-detail/$id/route";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import {
  SectionsGroup,
  SectionLayout,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { Badge } from "~/components/ui/badge";

function getFeedbackTypeBadgeStyle(type: FeedbackDetailData["feedback_type"]) {
  switch (type) {
    case "boolean":
      return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300";
    case "float":
      return "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300";
    case "comment":
      return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
    case "demonstration":
      return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300";
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}

function getFeedbackTypeLabel(type: FeedbackDetailData["feedback_type"]) {
  switch (type) {
    case "boolean":
      return "Boolean";
    case "float":
      return "Float";
    case "comment":
      return "Comment";
    case "demonstration":
      return "Demonstration";
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}

interface FeedbackDetailContentProps {
  data: FeedbackDetailData;
}

export function FeedbackDetailContent({ data }: FeedbackDetailContentProps) {
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Details" />
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Type</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Badge className={getFeedbackTypeBadgeStyle(data.feedback_type)}>
                {getFeedbackTypeLabel(data.feedback_type)}
              </Badge>
            </BasicInfoItemContent>
          </BasicInfoItem>
          <BasicInfoItem>
            <BasicInfoItemTitle>ID</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <code className="font-mono text-sm">{data.id}</code>
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </SectionLayout>
    </SectionsGroup>
  );
}
