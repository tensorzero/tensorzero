import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { ImageMessage } from "./SnippetContent";

async function getBase64File(url: string): Promise<string> {
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      resolve(base64String);
    };
    reader.readAsDataURL(blob);
  });
}

// Ferris the crab image from TensorZero tests
const ferrisImageUrl =
  "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png";

const meta = {
  title: "UI/Message Blocks/ImageMessage",
  component: ImageMessage,
  decorators: [withRouter],
  parameters: {
    layout: "padded",
  },
  argTypes: {
    downloadName: {
      control: "text",
      description: "Optional filename for downloads",
    },
  },
} satisfies Meta<typeof ImageMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    url: await getBase64File(ferrisImageUrl),
    downloadName: "ferris.png",
  },
};
