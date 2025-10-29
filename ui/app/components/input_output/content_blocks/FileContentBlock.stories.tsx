import { FileContentBlock } from "./FileContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import mp3Url from "./FileContentBlock.stories.fixture.mp3?url";
import pdfUrl from "./FileContentBlock.stories.fixture.pdf?url";

const meta = {
  title: "Input Output/Content Blocks/FileContentBlock",
  component: FileContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof FileContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

// TODO (GabrielBianconi): in the future this should be an Option<String> so we can handle failures more gracefully (or alternatively, another variant for `File`)
async function getBase64File(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      // TODO (GabrielBianconi): in the future this should be an Option<String> so we can handle failures more gracefully (or alternatively, another variant for `File`)
      return "";
    }
    const blob = await response.blob();
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        resolve(`data:${blob.type};base64,${base64String.split(",")[1]}`);
      };
      // TODO (GabrielBianconi): in the future this should be an Option<String> so we can handle failures more gracefully (or alternatively, another variant for `File`)
      reader.onerror = () => resolve("");
      reader.readAsDataURL(blob);
    });
  } catch {
    // TODO (GabrielBianconi): in the future this should be an Option<String> so we can handle failures more gracefully (or alternatively, another variant for `File`)
    return "";
  }
}

export const ImageURL: Story = {
  name: "Image (URL)",
  args: {
    block: {
      source_url:
        "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
      mime_type: "image/png",
      data: "",
      storage_path: {
        kind: {
          type: "s3_compatible",
          bucket_name: "tensorzero-e2e-test-images",
          region: "us-east-1",
          endpoint: null,
          allow_http: null,
        },
        path: "observability/files/e46e28c76498f7a7e935a502d3cd6f41052a76a6c6b0d8cda44e03fad8cc70f1.png",
      },
    },
  },
};

export const ImageBase64: Story = {
  name: "Image (Base64)",
  args: {
    block: {
      data: await getBase64File(
        "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
      ),
      source_url:
        "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
      mime_type: "image/png",
      storage_path: {
        kind: {
          type: "filesystem",
          path: "my_object_storage",
        },
        path: "observability/files/ferris_image_base64.png",
      },
    },
  },
};

export const AudioURL: Story = {
  name: "Audio (URL)",
  args: {
    block: {
      data: await getBase64File(mp3Url),
      mime_type: "audio/mp3",
      storage_path: {
        kind: {
          type: "filesystem",
          path: "my_audio_storage",
        },
        path: "observability/files/meeting_recording_url.mp3",
      },
    },
  },
};

export const AudioBase64: Story = {
  name: "Audio (Base64)",
  args: {
    block: {
      data: await getBase64File(mp3Url),
      mime_type: "audio/mp3",
      storage_path: {
        kind: {
          type: "s3_compatible",
          bucket_name: "tensorzero-audio",
          region: "us-west-2",
          endpoint: null,
          allow_http: null,
        },
        path: "observability/files/audio_sample_base64.mp3",
      },
    },
  },
};

export const PDFURL: Story = {
  name: "PDF (URL)",
  args: {
    block: {
      data: await getBase64File(pdfUrl),
      mime_type: "application/pdf",
      storage_path: {
        kind: {
          type: "s3_compatible",
          bucket_name: "tensorzero-documents",
          region: "us-east-1",
          endpoint: null,
          allow_http: null,
        },
        path: "observability/files/contract_url.pdf",
      },
    },
  },
};

export const PDFBase64: Story = {
  name: "PDF (Base64)",
  args: {
    block: {
      data: await getBase64File(pdfUrl),
      mime_type: "application/pdf",
      storage_path: {
        kind: {
          type: "filesystem",
          path: "my_document_storage",
        },
        path: "observability/files/document_base64.pdf",
      },
    },
  },
};

export const Error: Story = {
  args: {
    block: {
      // TODO (GabrielBianconi): in the future this should be an Option<String> so we can handle failures more gracefully (or alternatively, another variant for `File`)
      data: "",
      mime_type: "image/png",
      storage_path: {
        kind: {
          type: "s3_compatible",
          bucket_name: "tensorzero-e2e-test-images",
          region: "us-east-1",
          endpoint: null,
          allow_http: null,
        },
        path: "observability/files/failed_to_retrieve.png",
      },
    },
  },
};
