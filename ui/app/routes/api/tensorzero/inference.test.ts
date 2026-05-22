import { describe, expect, it } from "vitest";
import { action } from "./inference";

describe("/api/tensorzero/inference action", () => {
  it("returns 400 when the data form field contains malformed JSON", async () => {
    const formData = new FormData();
    formData.set("data", "{not json");

    const response = await action({
      request: new Request("http://localhost/api/tensorzero/inference", {
        method: "POST",
        body: formData,
      }),
    } as never);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Error parsing request data",
    });
  });

  it("returns 400 when the data form field is missing", async () => {
    const formData = new FormData();

    const response = await action({
      request: new Request("http://localhost/api/tensorzero/inference", {
        method: "POST",
        body: formData,
      }),
    } as never);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "Missing request data",
    });
  });
});
