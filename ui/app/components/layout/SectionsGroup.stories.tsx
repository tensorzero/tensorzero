import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import {
  SectionsGroup,
  SectionLayout,
  SectionHeader,
  PageLayout,
} from "./PageLayout";
import { Button } from "~/components/ui/button";
import { Card } from "~/components/ui/card";
import {
  Inferences,
  Episodes,
  Dataset,
  Functions,
} from "~/components/icons/Icons";

const meta = {
  title: "Layouts/SectionsGroup",
  component: SectionsGroup,
  decorators: [withRouter],
  parameters: {
    layout: "fullscreen",
  },
  render: (args) => (
    <PageLayout>
      <SectionsGroup {...args} />
    </PageLayout>
  ),
} satisfies Meta<typeof SectionsGroup>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  args: {
    children: (
      <>
        <SectionLayout>
          <SectionHeader heading="First Section" />
          <Card className="p-4">
            <p>Content of the first section.</p>
          </Card>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Second Section" />
          <Card className="p-4">
            <p>Content of the second section.</p>
          </Card>
        </SectionLayout>
      </>
    ),
  },
};

export const Dashboard: Story = {
  args: {
    children: (
      <>
        <SectionLayout>
          <SectionHeader heading="Overview" count={3} />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Card className="p-4">
              <div className="mb-2 flex items-center gap-2">
                <Inferences className="text-fg-secondary h-5 w-5" />
                <h3 className="font-medium">Inferences</h3>
              </div>
              <p className="text-2xl font-bold">1,234</p>
            </Card>
            <Card className="p-4">
              <div className="mb-2 flex items-center gap-2">
                <Episodes className="text-fg-secondary h-5 w-5" />
                <h3 className="font-medium">Episodes</h3>
              </div>
              <p className="text-2xl font-bold">567</p>
            </Card>
            <Card className="p-4">
              <div className="mb-2 flex items-center gap-2">
                <Dataset className="text-fg-secondary h-5 w-5" />
                <h3 className="font-medium">Datasets</h3>
              </div>
              <p className="text-2xl font-bold">89</p>
            </Card>
          </div>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader
            heading="Recent Activity"
            badge={{
              name: "Live",
              tooltip: "Updates in real-time",
            }}
          />
          <Card className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">New inference created</span>
                <span className="text-fg-secondary text-xs">2 min ago</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Dataset updated</span>
                <span className="text-fg-secondary text-xs">5 min ago</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Model training completed</span>
                <span className="text-fg-secondary text-xs">1 hour ago</span>
              </div>
            </div>
          </Card>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Actions" />
          <div className="flex flex-wrap gap-2">
            <Button>Create New Dataset</Button>
            <Button variant="outline">Run Evaluation</Button>
            <Button variant="outline">Export Data</Button>
            <Button variant="ghost">View Logs</Button>
          </div>
        </SectionLayout>
      </>
    ),
  },
};

export const Empty: Story = {
  args: {
    children: undefined,
  },
};

export const SingleSection: Story = {
  args: {
    children: (
      <SectionLayout>
        <SectionHeader heading="Only Section" />
        <Card className="p-4">
          <p>This SectionsGroup contains only one section.</p>
        </Card>
      </SectionLayout>
    ),
  },
};

export const ManySections: Story = {
  args: {
    children: (
      <>
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <SectionLayout key={i}>
            <SectionHeader heading={`Section ${i}`} count={i * 10} />
            <Card className="p-4">
              <p>Content for section {i}.</p>
            </Card>
          </SectionLayout>
        ))}
      </>
    ),
  },
};
