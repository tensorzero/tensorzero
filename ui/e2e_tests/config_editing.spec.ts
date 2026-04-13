import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { v7 as uuidv7 } from "uuid";

/**
 * Gateway URL for direct API calls used in test setup/teardown.
 * Uses the same env var as the UI server so both point at the same gateway.
 */
const GATEWAY_URL =
  process.env.TENSORZERO_GATEWAY_URL ?? "http://localhost:3000";

interface EditableConfig {
  toml: string;
  path_contents: Record<string, string>;
  base_signature: string;
  hash: string;
}

function filePathLabel(page: Page, path: string) {
  return page.getByText(path, { exact: true });
}

async function fetchLatestConfig(
  request: APIRequestContext,
): Promise<EditableConfig> {
  const response = await request.get(`${GATEWAY_URL}/internal/config_toml`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function applyConfig(
  request: APIRequestContext,
  config: Pick<EditableConfig, "toml" | "path_contents" | "base_signature">,
): Promise<EditableConfig> {
  const response = await request.post(
    `${GATEWAY_URL}/internal/config_toml/apply`,
    {
      data: config,
      headers: { "Content-Type": "application/json" },
    },
  );
  expect(
    response.ok(),
    `applyConfig failed: ${await response.text()}`,
  ).toBeTruthy();
  return response.json();
}

test.describe("Config Editor @config-editing", () => {
  test.describe.configure({ mode: "serial" });

  let originalConfig: EditableConfig;
  let firstFilePath: string;

  test.beforeAll(async ({ request }) => {
    originalConfig = await fetchLatestConfig(request);
    [firstFilePath] = Object.keys(originalConfig.path_contents).sort();
  });

  test.afterAll(async ({ request }) => {
    const latest = await fetchLatestConfig(request);
    await applyConfig(request, {
      toml: originalConfig.toml,
      path_contents: originalConfig.path_contents,
      base_signature: latest.base_signature,
    });
  });

  test("should load the config page", async ({ page }) => {
    await page.goto("/config");
    await page.waitForLoadState("networkidle");

    await expect(
      page.getByRole("heading", {
        name: "TensorZero Gateway Configuration",
      }),
    ).toBeVisible();
    await expect(page.getByText("Config TOML")).toBeVisible();
    await expect(page.getByText("Referenced Files")).toBeVisible();
  });

  test("can modify the gateway config TOML", async ({ page }) => {
    await page.goto("/config");
    await page.waitForLoadState("networkidle");

    const tomlEditor = page.locator(
      '[aria-label="Editable config TOML"] .cm-content',
    );
    await expect(tomlEditor).toBeVisible();

    // Insert a function at the start of the TOML
    await tomlEditor.click();
    await page.keyboard.press("ControlOrMeta+Home");
    await page.keyboard.type(
      '[embedding_models.dummy_playwright_embedding_model]\n\
      routing = ["dummy"]\n\
      [embedding_models.dummy_playwright_embedding_model.providers.dummy]\n\
      model_name = "test-embeddings"\n\
      type = "dummy"\n',
    );

    // Unsaved edits badge should appear
    await expect(page.getByText("Unsaved edits")).toBeVisible();

    // Save button should become enabled and visible
    const saveButton = page.getByRole("button", {
      name: "Save config and files",
    });
    await expect(saveButton).toBeEnabled();
    await saveButton.click();

    // Success toast
    await expect(page.getByText("Config applied").first()).toBeVisible();

    // Badge disappears after successful save
    await expect(page.getByText("Unsaved edits")).not.toBeVisible();

    // Reload and verify the config persisted
    await page.reload();
    await page.waitForLoadState("networkidle");
    await expect(tomlEditor).toBeVisible();
    const content = await tomlEditor.textContent();
    expect(content).toContain("dummy_playwright_embedding_model");
  });

  test("can modify a referenced file", async ({ page }) => {
    await page.goto("/config");
    await page.waitForLoadState("networkidle");

    // Click the first file in the sidebar
    await filePathLabel(page, firstFilePath).click();

    const fileEditor = page.locator(
      `[aria-label="Editable content for ${firstFilePath}"] .cm-content`,
    );
    await expect(fileEditor).toBeVisible();

    // Replace the file content
    await fileEditor.click();
    await page.keyboard.press("ControlOrMeta+A");

    const newFileContent = uuidv7();
    await page.keyboard.type(newFileContent);

    await page.getByRole("button", { name: "Save config and files" }).click();
    await expect(page.getByText("Config applied").first()).toBeVisible();

    // Reload and verify the new content persists
    await page.reload();
    await page.waitForLoadState("networkidle");
    await filePathLabel(page, firstFilePath).click();
    await expect(fileEditor).toBeVisible();
    const content = await fileEditor.textContent();
    expect(content).toContain(newFileContent);
  });

  test("can add a new file", async ({ page }) => {
    await page.goto("/config");
    await page.waitForLoadState("networkidle");

    // Click the New file button and rename it to a known path so we can
    // reference it by name in the TOML.
    await page.getByLabel("New file").click();
    await page.getByLabel("File name").fill("playwright-template-1");
    await page.getByRole("button", { name: "Rename", exact: true }).click();

    const newFilePath = "playwright-template-1";
    await expect(filePathLabel(page, newFilePath)).toBeVisible();

    // Add Jinja template content to the new file
    const fileEditor = page.locator(
      `[aria-label="Editable content for ${newFilePath}"] .cm-content`,
    );
    await expect(fileEditor).toBeVisible();
    await fileEditor.click();
    await page.keyboard.type("{{ user_text }}!");

    // Add a function with a variant that references the new file via
    // user_template so the file is included in the canonical config and
    // persists through the normalize round-trip on reload.
    const tomlEditor = page.locator(
      '[aria-label="Editable config TOML"] .cm-content',
    );
    await tomlEditor.click();
    await page.keyboard.press("ControlOrMeta+End");
    await page.keyboard.type(
      '\n\n[functions.playwright_edit_test]\ntool_choice = "auto"\ntools = []\ntype = "chat"\n[functions.playwright_edit_test.variants.dummy]\nmodel = "dummy::good"\nuser_template = "playwright-template-1"\ntype = "chat_completion"',
    );

    await page.getByRole("button", { name: "Save config and files" }).click();
    await expect(page.getByText("Config applied").first()).toBeVisible();

    // Reload and verify the file and its content persist
    await page.reload();
    await page.waitForLoadState("networkidle");
    await expect(filePathLabel(page, newFilePath)).toBeVisible();
    await filePathLabel(page, newFilePath).click();
    await expect(fileEditor).toBeVisible();
    const content = await fileEditor.textContent();
    expect(content).toContain("{{ user_text }}!");
  });

  test("can rename a file", async ({ page }) => {
    await page.goto("/config");
    await page.waitForLoadState("networkidle");

    const originalPath = "playwright-rename-source.txt";
    const renamedPath = "playwright-rename-target.txt";

    await page.getByLabel("New file").click();
    await page.getByLabel("File name").fill(originalPath);
    await page.getByRole("button", { name: "Rename", exact: true }).click();

    await expect(filePathLabel(page, originalPath)).toBeVisible();
    await expect(page.getByLabel("File name")).toHaveValue(originalPath);

    // Update the file name input and click Rename
    await page.getByLabel("File name").fill(renamedPath);
    await page.getByRole("button", { name: "Rename", exact: true }).click();

    // Sidebar should show the new name, not the old one
    await expect(filePathLabel(page, renamedPath)).toBeVisible();
    await expect(filePathLabel(page, originalPath)).not.toBeVisible();
    await expect(page.getByLabel("File name")).toHaveValue(renamedPath);

    await page.getByRole("button", { name: "Save config and files" }).click();
    await expect(page.getByText("Config applied").first()).toBeVisible();

    // Reload and verify the rename persisted
    await page.reload();
    await page.waitForLoadState("networkidle");
    await expect(filePathLabel(page, renamedPath)).toBeVisible();
    await expect(filePathLabel(page, originalPath)).not.toBeVisible();
  });
});
