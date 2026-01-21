import type { Meta, StoryObj } from "@storybook/react-vite";
import { Breadcrumbs } from "./Breadcrumbs";

const meta = {
  title: "Layout/Breadcrumbs",
  component: Breadcrumbs,
  decorators: [
    (Story) => (
      <div className="p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof Breadcrumbs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SingleSegment: Story = {
  args: {
    segments: [{ label: "Datasets", href: "/datasets" }],
  },
};

export const TwoSegments: Story = {
  args: {
    segments: [
      { label: "Datasets", href: "/datasets" },
      { label: "my-dataset", href: "/datasets/my-dataset", isIdentifier: true },
    ],
  },
};

export const WithNonClickableSegment: Story = {
  args: {
    segments: [
      { label: "Datasets", href: "/datasets" },
      { label: "my-dataset", href: "/datasets/my-dataset", isIdentifier: true },
      { label: "Datapoints" },
    ],
  },
};

export const FunctionVariant: Story = {
  args: {
    segments: [
      { label: "Functions", href: "/observability/functions" },
      {
        label: "extract_user_info",
        href: "/observability/functions/extract_user_info",
        isIdentifier: true,
      },
      { label: "Variants" },
    ],
  },
};

export const EvaluationResult: Story = {
  args: {
    segments: [
      { label: "Evaluations", href: "/evaluations" },
      {
        label: "quality_eval",
        href: "/evaluations/quality_eval",
        isIdentifier: true,
      },
      { label: "Results" },
    ],
  },
};
