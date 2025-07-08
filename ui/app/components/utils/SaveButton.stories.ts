import type { Meta, StoryObj } from "@storybook/react-vite";
import { fn } from "storybook/test";

import { SaveButton } from "./SaveButton";

const meta = {
  title: "SaveButton",
  component: SaveButton,
} satisfies Meta<typeof SaveButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Enabled: Story = {
  args: {
    disabled: false,
    onClick: fn(),
  },
};

export const Disabled: Story = {
  args: {
    disabled: true,
    onClick: fn(),
  },
};
