import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { SectionLayout, SectionHeader, PageLayout } from "./PageLayout";
import { Button } from "~/components/ui/button";
import { Card } from "~/components/ui/card";

const meta = {
  title: "Layouts/SectionLayout",
  component: SectionLayout,
  decorators: [withRouter],
  parameters: {
    layout: "fullscreen",
  },
  render: (args) => (
    <PageLayout>
      <SectionLayout {...args} />
    </PageLayout>
  ),
} satisfies Meta<typeof SectionLayout>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  args: {
    children: (
      <>
        <SectionHeader heading="Single Section" />
        <Card className="p-4">
          <p>This is content within a section layout.</p>
        </Card>
      </>
    ),
  },
};

export const WithMultipleChildren: Story = {
  args: {
    children: (
      <>
        <SectionHeader heading="Multiple Elements Section" />
        <Card className="p-4">
          <h3 className="mb-2 font-medium">First Element</h3>
          <p className="text-fg-secondary text-sm">Some content here.</p>
        </Card>
        <Card className="p-4">
          <h3 className="mb-2 font-medium">Second Element</h3>
          <p className="text-fg-secondary text-sm">More content here.</p>
        </Card>
        <div className="flex gap-2">
          <Button size="sm">Action 1</Button>
          <Button size="sm" variant="outline">
            Action 2
          </Button>
        </div>
      </>
    ),
  },
};

export const WithForm: Story = {
  args: {
    children: (
      <>
        <SectionHeader heading="Configuration Form" />
        <Card className="p-4">
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">Name</label>
              <input
                type="text"
                className="border-border w-full rounded-md border px-3 py-2"
                placeholder="Enter name"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">
                Description
              </label>
              <textarea
                className="border-border w-full rounded-md border px-3 py-2"
                rows={3}
                placeholder="Enter description"
              />
            </div>
          </div>
        </Card>
        <div className="flex gap-2">
          <Button>Save</Button>
          <Button variant="outline">Cancel</Button>
        </div>
      </>
    ),
  },
};

export const Empty: Story = {
  args: {
    children: undefined,
  },
};

export const OnlyHeader: Story = {
  args: {
    children: <SectionHeader heading="Section With Only Header" />,
  },
};
