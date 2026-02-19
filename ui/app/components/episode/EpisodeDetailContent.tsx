import type {
  StoredInference,
  FeedbackRow,
  FeedbackBounds,
} from "~/types/tensorzero";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import FeedbackTable, {
  FeedbackTableHeaders,
} from "~/components/feedback/FeedbackTable";
import { VariantLink } from "~/components/function/variant/VariantLink";
import {
  TableItemTime,
  TableItemFunction,
  TableItemShortUuid,
} from "~/components/ui/TableItems";
import { toFunctionUrl, toInferenceUrl } from "~/utils/urls";
import {
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import type { FeedbackActionData } from "~/routes/api/feedback/route";

export interface EpisodeDetailData {
  episode_id: string;
  inferences: StoredInference[];
  num_inferences: number;
  num_feedbacks: number;
  feedback: FeedbackRow[];
  feedback_bounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
}

interface EpisodeDetailContentProps {
  data: EpisodeDetailData;
  onFeedbackAdded?: (redirectUrl?: string) => void;
}

export function EpisodeDetailContent({
  data,
  onFeedbackAdded,
}: EpisodeDetailContentProps) {
  const {
    episode_id,
    inferences,
    num_inferences,
    num_feedbacks,
    feedback,
    feedback_bounds,
    latestFeedbackByMetric,
  } = data;

  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const humanFeedbackFetcher = useFetcherWithReset<FeedbackActionData>();
  const formError =
    humanFeedbackFetcher.state === "idle"
      ? (humanFeedbackFetcher.data?.error ?? null)
      : null;

  useEffect(() => {
    const currentState = humanFeedbackFetcher.state;
    const fetcherData = humanFeedbackFetcher.data;
    if (currentState === "idle" && fetcherData?.redirectTo) {
      setIsModalOpen(false);
      if (onFeedbackAdded) {
        onFeedbackAdded(fetcherData.redirectTo);
      } else {
        navigate(fetcherData.redirectTo);
      }
    }
  }, [
    humanFeedbackFetcher.data,
    humanFeedbackFetcher.state,
    navigate,
    onFeedbackAdded,
  ]);

  return (
    <>
      <ActionBar>
        <HumanFeedbackModal
          isOpen={isModalOpen}
          onOpenChange={(isOpen) => {
            if (humanFeedbackFetcher.state !== "idle") {
              return;
            }
            if (!isOpen) {
              humanFeedbackFetcher.reset();
            }
            setIsModalOpen(isOpen);
          }}
          trigger={<HumanFeedbackButton />}
        >
          <humanFeedbackFetcher.Form method="post" action="/api/feedback">
            <HumanFeedbackForm
              episodeId={episode_id}
              formError={formError}
              isSubmitting={
                humanFeedbackFetcher.state === "submitting" ||
                humanFeedbackFetcher.state === "loading"
              }
            />
          </humanFeedbackFetcher.Form>
        </HumanFeedbackModal>
        <AskAutopilotButton message={`Episode ID: ${episode_id}\n\n`} />
      </ActionBar>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Inferences" count={num_inferences} />
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Function</TableHead>
                <TableHead>Variant</TableHead>
                <TableHead>Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {inferences.length === 0 ? (
                <TableEmptyState message="No inferences found" />
              ) : (
                inferences.map((inference) => (
                  <TableRow
                    key={inference.inference_id}
                    id={inference.inference_id}
                  >
                    <TableCell className="max-w-[200px]">
                      <TableItemShortUuid
                        id={inference.inference_id}
                        link={toInferenceUrl(inference.inference_id)}
                      />
                    </TableCell>
                    <TableCell>
                      <TableItemFunction
                        functionName={inference.function_name}
                        functionType={inference.type}
                        link={toFunctionUrl(inference.function_name)}
                      />
                    </TableCell>
                    <TableCell>
                      <VariantLink
                        variantName={inference.variant_name}
                        functionName={inference.function_name}
                      >
                        <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300 hover:text-gray-500">
                          {inference.variant_name}
                        </code>
                      </VariantLink>
                    </TableCell>
                    <TableCell>
                      <TableItemTime timestamp={inference.timestamp} />
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader
            heading="Feedback"
            count={num_feedbacks}
            badge={{
              name: "episode",
              tooltip:
                "This table only includes episode-level feedback. To see inference-level feedback, open the detail page for that inference.",
            }}
          />
          {feedback.length === 0 ? (
            <Table>
              <FeedbackTableHeaders />
              <TableBody>
                <TableEmptyState message="No feedback found" />
              </TableBody>
            </Table>
          ) : (
            <FeedbackTable
              feedback={feedback}
              latestCommentId={feedback_bounds.by_type.comment.last_id!}
              latestDemonstrationId={
                feedback_bounds.by_type.demonstration.last_id!
              }
              latestFeedbackIdByMetric={latestFeedbackByMetric}
            />
          )}
        </SectionLayout>
      </SectionsGroup>
    </>
  );
}
