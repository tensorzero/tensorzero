import type { Meta, StoryObj } from "@storybook/react-vite";
import FeedbackBadges from "./FeedbackBadges";
import { Badge } from "~/components/ui/badge";
import type { FeedbackConfig } from "~/utils/config/feedback";

const meta = {
  title: "FeedbackBadges",
  component: FeedbackBadges,
} satisfies Meta<typeof FeedbackBadges>;

export default meta;
type Story = StoryObj<typeof meta>;

// All badge variant combinations
const booleanMaxInference: FeedbackConfig = {
  type: "boolean",
  optimize: "max",
  level: "inference",
};

const booleanMaxEpisode: FeedbackConfig = {
  type: "boolean",
  optimize: "max",
  level: "episode",
};

const booleanMinInference: FeedbackConfig = {
  type: "boolean",
  optimize: "min",
  level: "inference",
};

const booleanMinEpisode: FeedbackConfig = {
  type: "boolean",
  optimize: "min",
  level: "episode",
};

const floatMaxInference: FeedbackConfig = {
  type: "float",
  optimize: "max",
  level: "inference",
};

const floatMaxEpisode: FeedbackConfig = {
  type: "float",
  optimize: "max",
  level: "episode",
};

const floatMinInference: FeedbackConfig = {
  type: "float",
  optimize: "min",
  level: "inference",
};

const floatMinEpisode: FeedbackConfig = {
  type: "float",
  optimize: "min",
  level: "episode",
};

// Badge color classes - same as in FeedbackBadges.tsx
const badgeColors = {
  boolean: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300",
  float:
    "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
  demonstration:
    "bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-300",
  max: "bg-orange-200 text-orange-800 dark:bg-orange-800 dark:text-orange-200",
  min: "bg-amber-200 text-amber-700 dark:bg-amber-800 dark:text-amber-200",
  episode:
    "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  inference:
    "bg-yellow-200 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200",
};

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-6 p-4">
      <h2 className="text-lg font-semibold">All Badge Variants</h2>

      <div className="flex flex-col gap-4">
        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Boolean + Max + Inference
          </h3>
          <FeedbackBadges metric={booleanMaxInference} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Boolean + Max + Episode
          </h3>
          <FeedbackBadges metric={booleanMaxEpisode} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Boolean + Min + Inference
          </h3>
          <FeedbackBadges metric={booleanMinInference} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Boolean + Min + Episode
          </h3>
          <FeedbackBadges metric={booleanMinEpisode} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Float + Max + Inference
          </h3>
          <FeedbackBadges metric={floatMaxInference} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Float + Max + Episode
          </h3>
          <FeedbackBadges metric={floatMaxEpisode} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Float + Min + Inference
          </h3>
          <FeedbackBadges metric={floatMinInference} />
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Float + Min + Episode
          </h3>
          <FeedbackBadges metric={floatMinEpisode} />
        </div>
      </div>

      <h2 className="mt-4 text-lg font-semibold">Individual Badge Colors</h2>
      <p className="text-fg-muted text-sm">
        All badges use orange/amber/yellow spectrum for visual cohesion
      </p>

      <div className="flex flex-col gap-4">
        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Type Badges
          </h3>
          <div className="flex gap-2">
            <Badge className={badgeColors.boolean}>boolean</Badge>
            <Badge className={badgeColors.float}>float</Badge>
            <Badge className={badgeColors.demonstration}>demonstration</Badge>
          </div>
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Optimize Badges
          </h3>
          <div className="flex gap-2">
            <Badge className={badgeColors.max}>max</Badge>
            <Badge className={badgeColors.min}>min</Badge>
          </div>
        </div>

        <div>
          <h3 className="text-fg-muted mb-2 text-sm font-medium">
            Level Badges
          </h3>
          <div className="flex gap-2">
            <Badge className={badgeColors.episode}>episode</Badge>
            <Badge className={badgeColors.inference}>inference</Badge>
          </div>
        </div>
      </div>
    </div>
  ),
};

export const BooleanMaxInference: Story = {
  args: {
    metric: booleanMaxInference,
  },
};

export const FloatMinEpisode: Story = {
  args: {
    metric: floatMinEpisode,
  },
};
