import { test, expect } from "@playwright/test";

const pages = [
  {
    name: "function detail",
    url: "/observability/functions/extract_entities",
    waitFor: "Variants",
    expectedMessagePrefix: "Function: extract_entities",
  },
  {
    name: "variant detail",
    url: "/observability/functions/extract_entities/variants/dicl",
    waitFor: "k (Neighbors)",
    expectedMessagePrefix: "Variant: dicl",
  },
  {
    name: "inference detail",
    url: "/observability/inferences/0196367a-842d-74c2-9e62-67e058632503",
    waitFor: "0196367a-842d-74c2-9e62-67f07369b6ad",
    expectedMessagePrefix: "Inference ID: 0196367a-842d-74c2-9e62-67e058632503",
  },
  {
    name: "episode detail",
    url: "/observability/episodes/0196367a-842d-74c2-9e62-67f07369b6ad",
    waitFor: "tensorzero::llm_judge::haiku::topic_starts_with_f",
    expectedMessagePrefix: "Episode ID: 0196367a-842d-74c2-9e62-67f07369b6ad",
  },
  {
    name: "dataset detail",
    url: "/datasets/foo",
    waitFor: "Datapoints",
    expectedMessagePrefix: "Dataset: foo",
  },
  {
    name: "datapoint detail",
    url: "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
    waitFor: "Input",
    expectedMessagePrefix: "Datapoint ID: 0196374b-d575-77b3-ac22-91806c67745c",
  },
  {
    name: "evaluation detail",
    url: "/evaluations/entity_extraction?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063%2C0196367b-c0bb-7f90-b651-f90eb9fba8f3",
    waitFor: "Input",
    expectedMessagePrefix: "Evaluation: entity_extraction",
  },
];

for (const { name, url, waitFor, expectedMessagePrefix } of pages) {
  test(`Ask Autopilot button on ${name} navigates to new session with context`, async ({
    page,
  }) => {
    await page.goto(url);
    await expect(page.getByText(waitFor).first()).toBeVisible();

    const button = page.getByRole("button", { name: "Ask Autopilot" });
    await expect(button).toBeVisible();
    await button.click();

    await expect(page).toHaveURL(/\/autopilot\/sessions\/new\?message=/);
    const messageParam = decodeURIComponent(
      new URL(page.url()).searchParams.get("message") ?? "",
    );
    expect(
      messageParam,
      `message param should start with entity context for ${name}`,
    ).toContain(expectedMessagePrefix);
  });
}
