import { FileContentBlock } from "./FileContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { File, ObjectStorageFile } from "~/types/tensorzero";
import mp3Url from "./FileContentBlock.stories.fixture.mp3?url";
import pdfUrl from "./FileContentBlock.stories.fixture.pdf?url";
import { TooltipProvider } from "~/components/ui/tooltip";

const meta = {
  title: "Input Output/Content Blocks/FileContentBlock",
  component: FileContentBlock,
  excludeStories: ["getBase64File"],
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <TooltipProvider>
            <Story />
          </TooltipProvider>
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof FileContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Helper function for Storybook stories that fetches a file from a URL and returns
 * either an ObjectStorageFile (on success) or ObjectStorageError (on failure).
 * This allows stories to gracefully handle file loading errors instead of silently
 * failing with empty strings.
 */
export async function getBase64File(
  block: Omit<ObjectStorageFile, "data"> & { source_url: string },
): Promise<File> {
  try {
    const response = await fetch(block.source_url);
    if (!response.ok) {
      return {
        file_type: "object_storage_error",
        error: `Failed to fetch file: ${response.status} ${response.statusText}`,
        source_url: block.source_url,
        mime_type: block.mime_type,
        storage_path: block.storage_path,
      };
    }
    const blob = await response.blob();
    const base64Data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        resolve(`data:${blob.type};base64,${base64String.split(",")[1]}`);
      };
      reader.onerror = () => reject("Failed to read file");
      reader.readAsDataURL(blob);
    });

    return {
      file_type: "object_storage",
      data: base64Data,
      source_url: block.source_url,
      mime_type: block.mime_type,
      storage_path: block.storage_path,
    };
  } catch (err: unknown) {
    const errorMessage =
      err && typeof err === "object" && "message" in err
        ? String(err.message)
        : String(err) || "Unknown error occurred";
    return {
      file_type: "object_storage_error",
      error: errorMessage,
      source_url: block.source_url,
      mime_type: block.mime_type,
      storage_path: block.storage_path,
    };
  }
}

export const ImageObjectStorage: Story = {
  name: "Image (Object Storage)",
  args: {
    block: await getBase64File({
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
    }),
  },
};

export const AudioObjectStorage: Story = {
  name: "Audio (Object Storage)",
  args: {
    block: await getBase64File({
      source_url: mp3Url,
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
    }),
  },
};

export const PDFObjectStorage: Story = {
  name: "PDF (Object Storage)",
  args: {
    block: await getBase64File({
      source_url: pdfUrl,
      mime_type: "application/pdf",
      storage_path: {
        kind: {
          type: "filesystem",
          path: "my_document_storage",
        },
        path: "observability/files/document_base64.pdf",
      },
    }),
  },
};

export const Error: Story = {
  args: {
    block: {
      file_type: "object_storage_error",
      error: "You are not authorized to access this file.",
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

export const ErrorLong: Story = {
  name: "Error (Long)",
  args: {
    block: {
      file_type: "object_storage_error",
      error:
        "You are not authorized to access this file. Code: " + "01".repeat(100),
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

export const FileUrl: Story = {
  name: "File URL",
  args: {
    block: {
      file_type: "url",
      url: "https://example.com/image.png",
      mime_type: "image/png",
      filename: "example-image.png",
      detail: "high",
    },
  },
};

export const FileUrlEmpty: Story = {
  name: "File URL (Empty)",
  args: {
    block: {
      file_type: "url",
      url: "",
      mime_type: null,
    },
  },
};

export const FileUrlEditing: Story = {
  name: "File URL (Editing)",
  args: {
    block: {
      file_type: "url",
      url: "https://example.com/image.png",
      mime_type: "image/png",
      filename: "example-image.png",
      detail: "high",
    },
    isEditing: true,
  },
};

export const FileBase64: Story = {
  name: "File Base64",
  args: {
    block: {
      file_type: "base64",
      data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      mime_type: "image/png",
      filename: "pixel.png",
      detail: "auto",
    },
  },
};

export const FileBase64Empty: Story = {
  name: "File Base64 (Empty)",
  args: {
    block: {
      file_type: "base64",
      data: "",
      mime_type: "",
    },
  },
};

export const FileBase64Editing: Story = {
  name: "File Base64 (Editing)",
  args: {
    block: {
      file_type: "base64",
      data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      mime_type: "image/png",
      filename: "pixel.png",
      detail: "auto",
    },
    isEditing: true,
  },
};

export const FileBase64Long: Story = {
  name: "File Base64 (Long Data)",
  args: {
    block: {
      file_type: "base64",
      data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk".repeat(
        10,
      ),
      mime_type: "image/png",
      filename: "large-image.png",
    },
  },
};
