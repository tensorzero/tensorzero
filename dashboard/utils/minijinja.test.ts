import { describe, it, expect, beforeAll } from "vitest";
import { create_env, JsExposedEnv } from "./minijinja/pkg";

describe("Minijinja Wasm Integration", () => {
  let env: JsExposedEnv;

  beforeAll(async () => {
    await init(); // Initialize the Wasm module
    const templates = {
      greeting: "Hello, {{ name }}!",
      farewell: "Goodbye, {{ name }}.",
      invalid: "Hello, {{ invalid syntax }",
    };
    env = await create_env(templates);
  });

  it("renders greeting template correctly", async () => {
    const context = { name: "Alice" };
    const result = await env.render("greeting", JSON.stringify(context));
    expect(result).toBe("Hello, Alice!");
  });

  it("renders farewell template correctly", async () => {
    const context = { name: "Bob" };
    const result = await env.render("farewell", JSON.stringify(context));
    expect(result).toBe("Goodbye, Bob.");
  });

  it("handles invalid template gracefully", async () => {
    const invalidContext = { name: "Charlie" };
    await expect(
      env.render("invalid", JSON.stringify(invalidContext))
    ).rejects.toThrow();
  });
});
