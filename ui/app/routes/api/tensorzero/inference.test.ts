import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockGetExtraInferenceOptions, mockInference } = vi.hoisted(() => ({
  mockGetExtraInferenceOptions: vi.fn(() => ({})),
  mockInference: vi.fn(),
}));

vi.mock("~/utils/tensorzero.server", () => ({
  getTensorZeroClient: vi.fn(() => ({
    inference: mockInference,
  })),
}));

vi.mock("~/utils/feature_flags.server", () => ({
  getExtraInferenceOptions: mockGetExtraInferenceOptions,
}));

vi.mock("~/utils/logger", () => ({
  logger: {
    warn: vi.fn(),
  },
}));

import { action } from "./inference";

function makeRequest(data?: string): Request {
  const formData = new FormData();
  if (data !== undefined) {
    formData.set("data", data);
  }

  return new Request("http://localhost/api/tensorzero/inference", {
    method: "POST",
    body: formData,
  });
}

describe("tensorzero inference API action", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetExtraInferenceOptions.mockReturnValue({});
  });

  it("returns 400 when request data is missing", async () => {
    const response = await action({ request: makeRequest() } as never);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Missing request data",
    });
    expect(mockInference).not.toHaveBeenCalled();
  });

  it("returns 400 when request data is malformed JSON", async () => {
    const response = await action({ request: makeRequest("{") } as never);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Error parsing request data",
    });
    expect(mockInference).not.toHaveBeenCalled();
  });

  it("calls inference with parsed data and extra inference options", async () => {
    const inferenceResponse = {
      inference_id: "01942e26-4693-7e80-8591-47b98e25d721",
    };
    mockInference.mockResolvedValueOnce(inferenceResponse);
    mockGetExtraInferenceOptions.mockReturnValueOnce({
      internal: true,
    });

    const response = await action({
      request: makeRequest(
        JSON.stringify({
          function_name: "write_haiku",
          input: { messages: [] },
        }),
      ),
    } as never);

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(inferenceResponse);
    expect(mockInference).toHaveBeenCalledWith({
      function_name: "write_haiku",
      input: { messages: [] },
      internal: true,
    });
  });

  it("preserves existing TensorZero client error handling", async () => {
    mockInference.mockRejectedValueOnce(new Error("gateway unavailable"));

    const response = await action({
      request: makeRequest(
        JSON.stringify({
          function_name: "write_haiku",
          input: { messages: [] },
        }),
      ),
    } as never);

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual({
      error: "gateway unavailable",
    });
  });
});
