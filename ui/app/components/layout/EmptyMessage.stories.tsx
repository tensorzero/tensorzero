import type { Meta, StoryObj } from "@storybook/react-vite";
import { EmptyMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/EmptyMessage",
  component: EmptyMessage,
  parameters: {
    layout: "centered",
  },
} satisfies Meta<typeof EmptyMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const CustomMessage: Story = {
  args: {
    message: "No data available at this time",
  },
};

export const LongMessage: Story = {
  args: {
    message:
      "The requested information could not be found. Please check your filters and try again.",
  },
};
