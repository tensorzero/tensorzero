import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { useState } from "react";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";

const FF_ENABLE_FEEDBACK =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_FEEDBACK === "1";

interface EpisodeActionsProps {
  episodeId: string;
}

export function EpisodeActions({ episodeId }: EpisodeActionsProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleModalOpen = () => setIsModalOpen(true);
  const handleModalClose = () => setIsModalOpen(false);

  return (
    <ActionBar>
      {FF_ENABLE_FEEDBACK && <HumanFeedbackButton onClick={handleModalOpen} />}
      <HumanFeedbackModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        episodeId={episodeId}
      />
    </ActionBar>
  );
}
