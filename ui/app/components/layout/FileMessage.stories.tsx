import type { Meta, StoryObj } from "@storybook/react-vite";
import { FileMessage } from "./SnippetContent";
import pdfUrl from "../inference/Input.stories.fixture.tensorzero.pdf?url";

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
  title: "UI/Message Blocks/FileMessage",
  component: FileMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof FileMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const PDFFile: Story = {
  args: {
    fileData: await getBase64File(pdfUrl),
    filePath: "document.pdf",
    mimeType: "application/pdf",
  },
};

export const JSONFile: Story = {
  args: {
    fileData:
      "data:application/json;base64,eyJuYW1lIjoiSm9obiBEb2UiLCJhZ2UiOjMwLCJjaXR5IjoiTmV3IFlvcmsifQ==",
    filePath: "config.json",
    mimeType: "application/json",
  },
};

export const LongFilename: Story = {
  args: {
    fileData: await getBase64File(pdfUrl),
    filePath:
      "very_long_document_filename_that_should_be_truncated_properly_in_the_file_display_component.pdf",
    mimeType: "application/pdf",
  },
};
