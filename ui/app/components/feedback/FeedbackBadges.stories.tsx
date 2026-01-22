import type { Meta, StoryObj } from "@storybook/react-vite";
import FeedbackBadges, { getBadgeStyle } from "./FeedbackBadges";
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

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-8 p-6">
      {/* Color Palette */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">Badge Color Palette</h2>
        <div className="flex gap-12">
          <div className="flex flex-col items-start gap-2">
            <span className="text-fg-muted text-xs font-medium tracking-wide uppercase">
              Type
            </span>
            <div className="flex flex-col items-start gap-1.5">
              <Badge className={getBadgeStyle("type", "boolean")}>
                boolean
              </Badge>
              <Badge className={getBadgeStyle("type", "float")}>float</Badge>
              <Badge className={getBadgeStyle("type", "demonstration")}>
                demonstration
              </Badge>
            </div>
          </div>
          <div className="flex flex-col items-start gap-2">
            <span className="text-fg-muted text-xs font-medium tracking-wide uppercase">
              Optimize
            </span>
            <div className="flex flex-col items-start gap-1.5">
              <Badge className={getBadgeStyle("optimize", "max")}>max</Badge>
              <Badge className={getBadgeStyle("optimize", "min")}>min</Badge>
            </div>
          </div>
          <div className="flex flex-col items-start gap-2">
            <span className="text-fg-muted text-xs font-medium tracking-wide uppercase">
              Level
            </span>
            <div className="flex flex-col items-start gap-1.5">
              <Badge className={getBadgeStyle("level", "episode")}>
                episode
              </Badge>
              <Badge className={getBadgeStyle("level", "inference")}>
                inference
              </Badge>
            </div>
          </div>
        </div>
      </section>

      {/* Combinations Grid */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">Metric Combinations</h2>
        <div className="grid grid-cols-2 gap-x-8 gap-y-4">
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              boolean / max / inference
            </span>
            <FeedbackBadges metric={booleanMaxInference} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              boolean / max / episode
            </span>
            <FeedbackBadges metric={booleanMaxEpisode} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              boolean / min / inference
            </span>
            <FeedbackBadges metric={booleanMinInference} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              boolean / min / episode
            </span>
            <FeedbackBadges metric={booleanMinEpisode} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              float / max / inference
            </span>
            <FeedbackBadges metric={floatMaxInference} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">float / max / episode</span>
            <FeedbackBadges metric={floatMaxEpisode} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">
              float / min / inference
            </span>
            <FeedbackBadges metric={floatMinInference} />
          </div>
          <div className="flex items-center justify-between gap-4 rounded border p-3">
            <span className="text-fg-muted text-sm">float / min / episode</span>
            <FeedbackBadges metric={floatMinEpisode} />
          </div>
        </div>
      </section>
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
