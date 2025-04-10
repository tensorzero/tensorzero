import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";

const FF_ENABLE_FEEDBACK =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_FEEDBACK === "1";

export function EpisodeActions() {
  return (
    <ActionBar>
      {FF_ENABLE_FEEDBACK && <HumanFeedbackButton onClick={() => {}} />}
    </ActionBar>
  );
}
