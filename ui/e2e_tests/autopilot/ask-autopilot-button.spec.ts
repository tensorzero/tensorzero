import { test, expect, type Page } from "@playwright/test";

/**
 * Verifies the "Ask Autopilot" button is present on all detail pages
 * that should have it. This test guards against accidental removal
 * during refactors.
 */

async function expectAskAutopilotButton(page: Page) {
  const button = page.getByRole("button", { name: "Ask Autopilot" });
  await expect(
    button,
    "Ask Autopilot button should be visible on this page",
  ).toBeVisible();
}

test("should show Ask Autopilot button on function detail page", async ({
  page,
}) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on variant detail page", async ({
  page,
}) => {
  await page.goto("/observability/functions/extract_entities/variants/dicl");
  await expect(page.getByText("k (Neighbors)")).toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on inference detail page", async ({
  page,
}) => {
  await page.goto(
    "/observability/inferences/0196367a-842d-74c2-9e62-67e058632503",
  );
  await expect(
    page.getByText("0196367a-842d-74c2-9e62-67f07369b6ad"),
  ).toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on episode detail page", async ({
  page,
}) => {
  await page.goto(
    "/observability/episodes/0196367a-842d-74c2-9e62-67f07369b6ad",
  );
  await expect(
    page.getByText("tensorzero::llm_judge::haiku::topic_starts_with_f"),
  ).toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on dataset detail page", async ({
  page,
}) => {
  await page.goto("/datasets/foo");
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on datapoint detail page", async ({
  page,
}) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );
  await expect(page.getByText("Input")).toBeVisible();
  await expectAskAutopilotButton(page);
});

test("should show Ask Autopilot button on evaluation detail page", async ({
  page,
}) => {
  await page.goto(
    "/evaluations/entity_extraction?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063%2C0196367b-c0bb-7f90-b651-f90eb9fba8f3",
  );
  await expect(page.getByText("Input")).toBeVisible();
  await expectAskAutopilotButton(page);
});
