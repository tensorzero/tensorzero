import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { PageLayout } from "./PageLayout";
import { Button } from "~/components/ui/button";
import { Card } from "~/components/ui/card";

const meta = {
  title: "Layouts/PageLayout",
  component: PageLayout,
  decorators: [withRouter],
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof PageLayout>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: () => (
    <PageLayout>
      <h1 className="text-2xl font-medium">Basic Page Layout</h1>
      <p className="text-fg-secondary">
        This is a basic page layout with standard container, padding, and gap
        styling.
      </p>
    </PageLayout>
  ),
};

export const WithCustomClass: Story = {
  render: () => (
    <PageLayout className="bg-bg-secondary">
      <h1 className="text-2xl font-medium">Custom Background</h1>
      <p className="text-fg-secondary">
        This page layout has a custom background color applied via className.
      </p>
    </PageLayout>
  ),
};

export const WithMultipleChildren: Story = {
  render: () => (
    <PageLayout>
      <h1 className="text-2xl font-medium">Multiple Children Example</h1>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card className="p-4">
          <h3 className="font-medium">Card 1</h3>
          <p className="text-fg-secondary text-sm">Content goes here</p>
        </Card>
        <Card className="p-4">
          <h3 className="font-medium">Card 2</h3>
          <p className="text-fg-secondary text-sm">Content goes here</p>
        </Card>
        <Card className="p-4">
          <h3 className="font-medium">Card 3</h3>
          <p className="text-fg-secondary text-sm">Content goes here</p>
        </Card>
      </div>
      <Button>Action Button</Button>
    </PageLayout>
  ),
};

export const WithComplexLayout: Story = {
  render: () => (
    <PageLayout>
      <div className="mx-auto flex w-full max-w-240 flex-col gap-12">
        <h1 className="text-2xl font-medium">Dashboard</h1>
        <div className="grid w-full grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
          <div className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Section 1</h2>
            <Card className="p-4">
              <p>Complex nested layout example</p>
            </Card>
          </div>
          <div className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Section 2</h2>
            <Card className="p-4">
              <p>More complex content</p>
            </Card>
          </div>
          <div className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Section 3</h2>
            <Card className="p-4">
              <p>Even more content</p>
            </Card>
          </div>
        </div>
      </div>
    </PageLayout>
  ),
};

export const Empty: Story = {
  render: () => <PageLayout />,
};
