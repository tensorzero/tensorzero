import { describe, expect, test } from "vitest";
import { getEvalRunIds } from "./evaluation.server";

describe("getEvalRunIds", () => {
  test("should return correct run ids for entity_extraction eval", async () => {
    const runIds = await getEvalRunIds("entity_extraction");
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
      {
        eval_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
    ]);
  });

  test("should return correct run ids for haiku eval", async () => {
    const runIds = await getEvalRunIds("haiku");
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
        variant_name: "initial_prompt_haiku_3_5",
      },
      {
        eval_run_id: "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        variant_name: "initial_prompt_gpt4o_mini",
      },
    ]);
  });
});
