import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { Switch, SwitchSize } from "./switch";

const meta: Meta<typeof Switch> = {
  title: "Switch",
  component: Switch,
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof Switch>;

function SwitchWithState({
  defaultChecked = false,
  ...props
}: Omit<React.ComponentProps<typeof Switch>, "checked" | "onCheckedChange"> & {
  defaultChecked?: boolean;
}) {
  const [checked, setChecked] = useState(defaultChecked);
  return <Switch {...props} checked={checked} onCheckedChange={setChecked} />;
}

export const Default: Story = {
  render: () => <SwitchWithState />,
};

export const Checked: Story = {
  render: () => <SwitchWithState defaultChecked />,
};

export const Small: Story = {
  render: () => <SwitchWithState size={SwitchSize.Small} />,
};

export const SmallChecked: Story = {
  render: () => <SwitchWithState size={SwitchSize.Small} defaultChecked />,
};

export const Medium: Story = {
  render: () => <SwitchWithState size={SwitchSize.Medium} />,
};

export const MediumChecked: Story = {
  render: () => <SwitchWithState size={SwitchSize.Medium} defaultChecked />,
};

export const Disabled: Story = {
  render: () => <SwitchWithState disabled />,
};

export const DisabledChecked: Story = {
  render: () => <SwitchWithState disabled defaultChecked />,
};

export const WithLabel: Story = {
  render: () => (
    <label className="flex cursor-pointer items-center gap-2">
      <span className="text-sm">Enable feature</span>
      <SwitchWithState />
    </label>
  ),
};
