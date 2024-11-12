import { describe, it, expect, beforeAll } from "vitest";
import { create_env, JsExposedEnv } from "./minijinja/pkg";

describe("Minijinja Wasm Integration", () => {
  let env: JsExposedEnv;

  beforeAll(async () => {
    const templates = {
      greeting: "Hello, {{ name }}!",
      farewell: "Goodbye, {{ name }}.",
    };
    env = await create_env(templates);
  });

  it("renders greeting template correctly", async () => {
    const context = { name: "Alice" };
    const result = await env.render("greeting", context);
    expect(result).toBe("Hello, Alice!");
  });

  it("renders farewell template correctly", async () => {
    const context = { name: "Bob" };
    const result = await env.render("farewell", context);
    expect(result).toBe("Goodbye, Bob.");
  });

  it("handles invalid template gracefully", async () => {
    const invalidContext = { name: "Charlie" };
    await expect(
      async () => await env.render("invalid", invalidContext),
    ).rejects.toThrow(/template not found: template "invalid" does not exist/i);
  });
});
