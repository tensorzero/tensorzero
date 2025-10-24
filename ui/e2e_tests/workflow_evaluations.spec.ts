import { test, expect } from "@playwright/test";

test("should show the workflow evaluations page and navigate to the workflow evaluation run page", async ({
  page,
}) => {
  await page.goto("/workflow_evaluations");
  await expect(page.getByText("Evaluation Runs")).toBeVisible();
  await expect(
    page.getByText("01968d04-142c-7e53-8ea7-3a3255b518dc"),
  ).toBeVisible();
  // Let's click on that run and see the run page
  await page.getByText("01968d04-142c-7e53-8ea7-3a3255b518dc").click();
  await expect(page.getByText("Workflow Evaluation Run")).toBeVisible();
  await expect(page.getByText("goated")).toBeVisible();
});

test("should render comment in workflow evaluation run and open modal when clicked", async ({
  page,
}) => {
  await page.goto(
    "/workflow_evaluations/runs/01968d05-d734-7751-ab33-75dd8b3fb4a3",
  );

  // Wait for the page to load
  await expect(page.getByText("Workflow Evaluation Run")).toBeVisible();

  // Check that the comment starting with "This comment is longer than a" is rendered in the table
  const commentText = page.getByText("This comment is longer than a");
  await expect(commentText).toBeVisible();

  // Click on the comment to open the modal
  await commentText.click();

  // Get the sheet/modal container
  const modal = page.locator('[role="dialog"]');

  // Wait for the sheet to animate open and verify the full comment content is visible in the modal
  await expect(
    modal.getByText("Boy, I hope the front-end devs can handle this!"),
  ).toBeVisible({ timeout: 5000 });

  // Now verify the other parts of the full comment are visible in the modal
  await expect(modal.getByText("It also has multiple lines.")).toBeVisible();
  await expect(
    modal.getByText(
      "This comment is longer than a typical column width would be.",
    ),
  ).toBeVisible();
});
