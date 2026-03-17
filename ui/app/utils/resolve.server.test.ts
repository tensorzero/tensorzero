import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGetObject = vi.fn<(storagePath: unknown) => Promise<string>>();

vi.mock("~/utils/tensorzero.server", () => ({
  getTensorZeroClient: vi.fn(() => ({
    getObject: mockGetObject,
  })),
}));

import {
  loadFileDataForInput,
  loadFileDataForStoredInput,
} from "./resolve.server";

describe("resolve.server", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads object-stored audio into canonical raw base64 for replay", async () => {
    mockGetObject.mockResolvedValueOnce(
      JSON.stringify({ data: "GkXfo59ChoEBQveBAULygQ==" }),
    );

    const input = {
      system: undefined,
      messages: [
        {
          role: "user" as const,
          content: [
            {
              type: "text" as const,
              text: "transcribe this",
            },
            {
              type: "file" as const,
              file_type: "object_storage_pointer" as const,
              mime_type: "audio/webm;codecs=opus",
              storage_path: {
                kind: {
                  type: "filesystem" as const,
                  path: "/tmp/object_storage",
                },
                path: "observability/files/audio_sample.webm",
              },
              filename: "audio_sample.webm",
            },
          ],
        },
      ],
    };

    const resolved = await loadFileDataForInput(input);
    const file = resolved.messages[0]?.content[1];

    expect(mockGetObject).toHaveBeenCalledTimes(1);
    expect(file).toMatchObject({
      type: "file",
      file_type: "object_storage",
      mime_type: "audio/webm;codecs=opus",
      data: "GkXfo59ChoEBQveBAULygQ==",
      filename: "audio_sample.webm",
    });
    expect((file as { data: string }).data).not.toMatch(/^data:/);
  });

  it("loads stored inputs without converting object storage bytes into a data URL", async () => {
    mockGetObject.mockResolvedValueOnce(
      JSON.stringify({ data: "iVBORw0KGgoAAAANSUhEUg==" }),
    );

    const storedInput = {
      system: undefined,
      messages: [
        {
          role: "user" as const,
          content: [
            {
              type: "file" as const,
              mime_type: "image/png",
              storage_path: {
                kind: {
                  type: "filesystem" as const,
                  path: "/tmp/object_storage",
                },
                path: "observability/files/image.png",
              },
              filename: "image.png",
            },
          ],
        },
      ],
    };

    const resolved = await loadFileDataForStoredInput(storedInput);
    const file = resolved.messages[0]?.content[0];

    expect(mockGetObject).toHaveBeenCalledTimes(1);
    expect(file).toMatchObject({
      type: "file",
      file_type: "object_storage",
      mime_type: "image/png",
      data: "iVBORw0KGgoAAAANSUhEUg==",
      filename: "image.png",
    });
    expect((file as { data: string }).data).not.toMatch(/^data:/);
  });
});
