import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { PageHeader, PageLayout } from "./PageLayout";
import { Button } from "~/components/ui/button";
import {
  Inferences,
  Episodes,
  Functions,
  Dataset,
} from "~/components/icons/Icons";

const meta = {
  title: "Layouts/PageHeader",
  component: PageHeader,
  decorators: [withRouter],
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof PageHeader>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: () => (
    <PageLayout>
      <PageHeader heading="Basic Page Header" />
    </PageLayout>
  ),
};

export const WithLabel: Story = {
  render: () => (
    <PageLayout>
      <PageHeader label="Dashboard" heading="Overview" />
    </PageLayout>
  ),
};

export const WithLabelAndIcon: Story = {
  render: () => (
    <PageLayout>
      <PageHeader
        label="Observability"
        heading="Inferences"
        icon={<Inferences className="h-4 w-4" />}
        iconBg="bg-blue-100"
      />
    </PageLayout>
  ),
};

export const WithCount: Story = {
  render: () => (
    <PageLayout>
      <PageHeader heading="Datasets" count={42} />
    </PageLayout>
  ),
};

export const WithLargeCount: Story = {
  render: () => (
    <PageLayout>
      <PageHeader heading="Inferences" count={1234567} />
    </PageLayout>
  ),
};

export const WithName: Story = {
  render: () => (
    <PageLayout>
      <PageHeader label="Datapoint" name="abc123-def456-789" lateral="Custom" />
    </PageLayout>
  ),
};

export const WithLongName: Story = {
  render: () => (
    <PageLayout>
      <PageHeader
        label="Episode"
        name="very-long-identifier-name-that-might-cause-layout-issues-in-some-scenarios"
        lateral="Automated"
      />
    </PageLayout>
  ),
};

export const WithChildren: Story = {
  render: () => (
    <PageLayout>
      <PageHeader heading="Page with Actions">
        <div className="flex flex-col gap-4 sm:flex-row">
          <Button>Create New</Button>
          <Button variant="outline">Import</Button>
          <Button variant="outline">Export</Button>
        </div>
      </PageHeader>
    </PageLayout>
  ),
};

export const Complex: Story = {
  render: () => (
    <PageLayout>
      <PageHeader
        label="Observability"
        heading="Functions"
        count={1234567}
        icon={<Functions className="h-4 w-4" />}
        iconBg="bg-green-100"
      >
        <div className="flex gap-2">
          <Button size="sm">Primary Action</Button>
          <Button size="sm" variant="outline">
            Secondary Action
          </Button>
        </div>
      </PageHeader>
    </PageLayout>
  ),
};

export const AllIcons: Story = {
  render: () => (
    <PageLayout>
      <div className="space-y-8">
        <PageHeader
          label="Observability"
          heading="Inferences"
          count={42}
          icon={<Inferences className="h-4 w-4" />}
          iconBg="bg-blue-100"
        />
        <PageHeader
          label="Observability"
          heading="Episodes"
          count={156}
          icon={<Episodes className="h-4 w-4" />}
          iconBg="bg-purple-100"
        />
        <PageHeader
          label="Observability"
          heading="Functions"
          count={8}
          icon={<Functions className="h-4 w-4" />}
          iconBg="bg-green-100"
        />
        <PageHeader
          label="Workflows"
          heading="Datasets"
          count={23}
          icon={<Dataset className="h-4 w-4" />}
          iconBg="bg-orange-100"
        />
      </div>
    </PageLayout>
  ),
};

export const LongText: Story = {
  render: () => (
    <PageLayout>
      <PageHeader
        label="Very Long Label That Might Wrap Or Be Truncated In Some Cases"
        heading="An Extremely Long Heading That Tests How The Layout Handles Very Long Text Content That Might Overflow"
        count={999999999}
        name="very-long-identifier-name-that-might-cause-layout-issues-in-some-scenarios"
        lateral="Very Long Lateral Text That Could Cause Issues"
      />
    </PageLayout>
  ),
};

export const Empty: Story = {
  render: () => (
    <PageLayout>
      <PageHeader />
    </PageLayout>
  ),
};
