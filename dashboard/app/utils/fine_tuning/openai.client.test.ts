import { describe, it, expect } from "vitest";
import { OpenAISFTJob } from "./openai.client";

describe("format_url", () => {
  it("formats URL with no path arg and no query params", () => {
    const job = new OpenAISFTJob("", "pending", undefined);
    const result = job.format_url("https://api.example.com");
    expect(result).toBe("https://api.example.com");
  });

  it("formats URL with path arg but no query params", () => {
    const job = new OpenAISFTJob("jobs/123", "pending", undefined);
    const result = job.format_url("https://api.example.com");
    expect(result).toBe("https://api.example.com/jobs/123");
  });

  it("handles base URLs with trailing slash", () => {
    const job = new OpenAISFTJob("jobs/123", "pending", undefined);
    const result = job.format_url("https://api.example.com/");
    expect(result).toBe("https://api.example.com//jobs/123");
  });
});
