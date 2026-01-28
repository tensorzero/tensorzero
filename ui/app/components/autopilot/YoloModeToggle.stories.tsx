import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { YoloModeToggle } from "./YoloModeToggle";

const meta: Meta<typeof YoloModeToggle> = {
  title: "Autopilot/YoloModeToggle",
  component: YoloModeToggle,
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof YoloModeToggle>;

function YoloModeToggleWithState({
  defaultChecked = false,
}: {
  defaultChecked?: boolean;
}) {
  const [checked, setChecked] = useState(defaultChecked);
  return <YoloModeToggle checked={checked} onCheckedChange={setChecked} />;
}

export const Default: Story = {
  render: () => <YoloModeToggleWithState />,
};

export const Enabled: Story = {
  render: () => <YoloModeToggleWithState defaultChecked />,
};

export const InHeader: Story = {
  render: () => (
    <div className="flex w-[600px] items-start justify-between rounded border p-4">
      <div>
        <h1 className="text-lg font-semibold">Session Page Header</h1>
        <p className="text-fg-muted text-sm">Session ID goes here</p>
      </div>
      <YoloModeToggleWithState />
    </div>
  ),
};

export const InHeaderEnabled: Story = {
  render: () => (
    <div className="flex w-[600px] items-start justify-between rounded border p-4">
      <div>
        <h1 className="text-lg font-semibold">Session Page Header</h1>
        <p className="text-fg-muted text-sm">Session ID goes here</p>
      </div>
      <YoloModeToggleWithState defaultChecked />
    </div>
  ),
};
