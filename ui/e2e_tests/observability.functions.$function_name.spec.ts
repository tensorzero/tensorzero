import { test, expect } from "@playwright/test";
import { DEFAULT_FUNCTION } from "~/utils/constants";

test("should show the function detail page", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByText("Variants")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should show description of chat function", async ({ page }) => {
  await page.goto("/observability/functions/write_haiku");
  await expect(page.getByText("Variants")).toBeVisible();
  await expect(
    page.getByText("Generate a haiku about a given topic"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should show description of json function", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByText("Variants")).toBeVisible();
  await expect(
    page.getByText("Extract named entities from text"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should show the function detail page for default function", async ({
  page,
}) => {
  await page.goto(`/observability/functions/${DEFAULT_FUNCTION}`);

  await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

const toolSearchWikipedia = {
  name: "search_wikipedia",
  description:
    "Search Wikipedia for pages that match the query. Returns a list of page titles.",
};

test("should open drawer when tool name is clicked", async ({ page }) => {
  await page.goto("/observability/functions/multi_hop_rag_agent");

  // ensure the drawer is not open by default
  const drawer = await page.getByRole("dialog");
  await expect(drawer).not.toBeVisible();

  // open the drawer
  const toolButton = await page.getByText(toolSearchWikipedia.name);
  await toolButton.click();
  await expect(drawer).toBeVisible();

  // ensure that the drawer shows the correct description
  const desc = await page.getByText(toolSearchWikipedia.description);
  await expect(desc.first()).toBeVisible();
});
