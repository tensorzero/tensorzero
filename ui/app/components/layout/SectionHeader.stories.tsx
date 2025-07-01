import type { Meta, StoryObj } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { SectionHeader, PageLayout } from "./PageLayout";
import { Button } from "~/components/ui/button";

const meta = {
  title: "Layouts/SectionHeader",
  component: SectionHeader,
  decorators: [withRouter],
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof SectionHeader>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="Basic Section" />
    </PageLayout>
  ),
};

export const WithCount: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="Results" count={156} />
    </PageLayout>
  ),
};

export const WithLargeCount: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="Inferences" count={12567890} />
    </PageLayout>
  ),
};

export const WithBadge: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader
        heading="Experimental Feature"
        badge={{
          name: "Beta",
          tooltip: "This feature is in beta and may change in future versions",
        }}
      />
    </PageLayout>
  ),
};

export const WithLongBadgeTooltip: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader
        heading="Advanced Analytics"
        badge={{
          name: "Pro",
          tooltip:
            "This is an extremely long tooltip text that explains a complex feature with lots of details that might wrap to multiple lines and test the tooltip rendering behavior in various scenarios.",
        }}
      />
    </PageLayout>
  ),
};

export const WithCountAndBadge: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader
        heading="AI Models"
        count={12}
        badge={{
          name: "New",
          tooltip: "Recently added models with enhanced capabilities",
        }}
      />
    </PageLayout>
  ),
};

export const WithChildren: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="Configuration">
        <Button size="sm" variant="ghost">
          Edit
        </Button>
      </SectionHeader>
    </PageLayout>
  ),
};

export const WithMultipleChildren: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="Actions">
        <div className="flex gap-2">
          <Button size="sm">Primary</Button>
          <Button size="sm" variant="outline">
            Secondary
          </Button>
          <Button size="sm" variant="ghost">
            Tertiary
          </Button>
        </div>
      </SectionHeader>
    </PageLayout>
  ),
};

export const ComplexExample: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader
        heading="Model Performance"
        count={156}
        badge={{
          name: "Live",
          tooltip: "Updates in real-time",
        }}
      >
        <div className="flex gap-2">
          <Button size="sm">Refresh</Button>
          <Button size="sm" variant="outline">
            Export
          </Button>
        </div>
      </SectionHeader>
    </PageLayout>
  ),
};

export const VariousBadges: Story = {
  render: () => (
    <PageLayout>
      <div className="space-y-6">
        <SectionHeader
          heading="Beta Features"
          count={3}
          badge={{
            name: "Beta",
            tooltip: "Features in beta testing",
          }}
        />
        <SectionHeader
          heading="Premium Models"
          count={8}
          badge={{
            name: "Pro",
            tooltip: "Available with Pro subscription",
          }}
        />
        <SectionHeader
          heading="Real-time Data"
          count={1245}
          badge={{
            name: "Live",
            tooltip: "Updates automatically",
          }}
        />
        <SectionHeader
          heading="Deprecated APIs"
          count={2}
          badge={{
            name: "Deprecated",
            tooltip: "Will be removed in future versions",
          }}
        />
        <SectionHeader
          heading="Latest Updates"
          count={15}
          badge={{
            name: "New",
            tooltip: "Recently added features",
          }}
        />
      </div>
    </PageLayout>
  ),
};

export const LongText: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader
        heading="Section With Extremely Long Heading That Tests Text Wrapping And Layout Behavior"
        count={123456789}
        badge={{
          name: "Very Long Badge Name",
          tooltip:
            "This is an extremely long tooltip text that explains a complex feature with lots of details that might wrap to multiple lines and test the tooltip rendering behavior in various scenarios.",
        }}
      />
    </PageLayout>
  ),
};

export const Empty: Story = {
  render: () => (
    <PageLayout>
      <SectionHeader heading="" />
    </PageLayout>
  ),
};
