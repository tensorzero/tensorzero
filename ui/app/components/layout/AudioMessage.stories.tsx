import type { Meta, StoryObj } from "@storybook/react-vite";
import { AudioMessage } from "./SnippetContent";
import mp3Url from "../inference/Input.stories.fixture.tensorzero.mp3?url";

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

const meta = {
  title: "UI/Message Blocks/AudioMessage",
  component: AudioMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof AudioMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    fileData: await getBase64File(mp3Url),
    filePath: "audio.mp3",
    mimeType: "audio/mp3",
  },
};

export const LongFilename: Story = {
  args: {
    fileData: await getBase64File(mp3Url),
    filePath:
      "very_long_audio_filename_that_should_be_truncated_properly_in_the_display.mp3",
    mimeType: "audio/mp3",
  },
};
