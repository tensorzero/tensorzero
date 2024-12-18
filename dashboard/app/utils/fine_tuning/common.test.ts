import { describe, it, expect } from "vitest";
import { OpenAISFTJob } from "./openai.client";
import { FireworksSFTJob } from "./fireworks.client";

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

  it("formats URL with query params but no path arg", () => {
    const job = new FireworksSFTJob("job123", "pending", undefined, undefined);
    const result = job.format_url("https://api.example.com");
    expect(result).toBe("https://api.example.com?jobPath=job123");
  });

  it("formats URL with both path arg and query params", () => {
    const job = new FireworksSFTJob("job123", "pending", "model456", undefined);
    const result = job.format_url("https://api.example.com");
    expect(result).toBe(
      "https://api.example.com?jobPath=job123&modelId=model456",
    );
  });

  it("handles base URLs with trailing slash", () => {
    const job = new OpenAISFTJob("jobs/123", "pending", undefined);
    const result = job.format_url("https://api.example.com/");
    expect(result).toBe("https://api.example.com//jobs/123");
  });

  it("properly encodes query parameters", () => {
    const job = new FireworksSFTJob(
      "job with spaces/123!@#",
      "pending",
      "model with spaces/456!@#",
      undefined,
    );
    const result = job.format_url("https://api.example.com");
    expect(result).toBe(
      "https://api.example.com?jobPath=job+with+spaces%2F123%21%40%23&modelId=model+with+spaces%2F456%21%40%23",
    );
  });
});
