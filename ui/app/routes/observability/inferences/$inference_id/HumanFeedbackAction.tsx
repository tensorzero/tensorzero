import { useState, useEffect, useCallback } from "react";
import type { StoredInference } from "~/types/tensorzero";
import type { FeedbackActionData } from "~/routes/api/feedback/route";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";

interface HumanFeedbackActionProps {
  inference: StoredInference;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

export function HumanFeedbackAction({
  inference,
  onFeedbackAdded,
}: HumanFeedbackActionProps) {
  const [isOpen, setIsOpen] = useState(false);

  const humanFeedbackFetcher = useFetcherWithReset<FeedbackActionData>();
  const {
    data: humanFeedbackData,
    state: humanFeedbackState,
    reset: resetHumanFeedbackFetcher,
  } = humanFeedbackFetcher;

  const formError =
    humanFeedbackState === "idle" ? (humanFeedbackData?.error ?? null) : null;

  useEffect(() => {
    if (humanFeedbackState === "idle" && humanFeedbackData?.redirectTo) {
      onFeedbackAdded(humanFeedbackData.redirectTo);
      setIsOpen(false);
      resetHumanFeedbackFetcher();
    }
  }, [
    humanFeedbackData,
    humanFeedbackState,
    onFeedbackAdded,
    resetHumanFeedbackFetcher,
  ]);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (humanFeedbackState !== "idle") return;
      if (!open) resetHumanFeedbackFetcher();
      setIsOpen(open);
    },
    [humanFeedbackState, resetHumanFeedbackFetcher],
  );

  return (
    <HumanFeedbackModal
      onOpenChange={handleOpenChange}
      isOpen={isOpen}
      trigger={<HumanFeedbackButton />}
    >
      <humanFeedbackFetcher.Form method="post" action="/api/feedback">
        <HumanFeedbackForm
          inferenceId={inference.inference_id}
          inferenceOutput={inference.output}
          formError={formError}
          isSubmitting={
            humanFeedbackState === "submitting" ||
            humanFeedbackState === "loading"
          }
        />
      </humanFeedbackFetcher.Form>
    </HumanFeedbackModal>
  );
}
