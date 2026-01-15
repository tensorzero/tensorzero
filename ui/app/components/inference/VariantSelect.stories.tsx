import type { Meta, StoryObj } from "@storybook/react-vite";
import { VariantSelect } from "./VariantSelect";

const mockVariants = [
  "gpt4o_variant",
  "gpt4o_mini_variant",
  "claude_variant",
  "llama_variant",
  "gemini_variant",
];

const mockModels = [
  "gpt-4o",
  "gpt-4o-mini",
  "claude-3-5-sonnet-latest",
  "claude-3-5-haiku-latest",
  "gemini-2.0-flash",
];

const manyVariants = Array.from(
  { length: 50 },
  (_, i) => `variant_${String(i + 1).padStart(3, "0")}`,
);

const meta = {
  title: "Inference/VariantSelect",
  component: VariantSelect,
  argTypes: {
    isLoading: {
      control: "boolean",
      description: "Whether the component is in a loading state",
    },
    isDefaultFunction: {
      control: "boolean",
      description:
        "Whether this is a default function (shows models instead of variants)",
    },
  },
  parameters: {
    controls: {
      exclude: ["onSelect", "options"],
    },
  },
  args: {
    onSelect: () => {},
    isLoading: false,
  },
} satisfies Meta<typeof VariantSelect>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    options: mockVariants,
    isDefaultFunction: false,
  },
};

export const DefaultFunction: Story = {
  args: {
    options: mockModels,
    isDefaultFunction: true,
  },
};

export const Loading: Story = {
  args: {
    options: mockVariants,
    isLoading: true,
  },
};

export const ManyVariants: Story = {
  args: {
    options: manyVariants,
    isDefaultFunction: false,
  },
};

export const SingleVariant: Story = {
  args: {
    options: ["only_variant"],
    isDefaultFunction: false,
  },
};

export const EmptyVariants: Story = {
  args: {
    options: [],
    isDefaultFunction: false,
  },
};
