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
