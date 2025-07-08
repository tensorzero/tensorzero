import type { Meta, StoryObj } from "@storybook/react-vite";
import { FileErrorMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/FileErrorMessage",
  component: FileErrorMessage,
  parameters: {
    layout: "padded",
  },
  argTypes: {
    error: {
      control: "text",
      description: "Error message to display",
    },
  },
} satisfies Meta<typeof FileErrorMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const FileNotFound: Story = {
  args: {
    error: "File not found",
  },
};

export const NetworkError: Story = {
  args: {
    error: "Network timeout",
  },
};

export const LongErrorMessage: Story = {
  args: {
    error:
      "Authentication failed: The access token has expired and the system was unable to refresh it automatically. Please log in again to continue accessing this resource.",
  },
};
