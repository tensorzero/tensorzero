import type { Meta, StoryObj } from "@storybook/react-vite";
import { DeleteButton } from "./DeleteButton";

const meta: Meta<typeof DeleteButton> = {
  title: "DeleteButton",
  component: DeleteButton,
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof DeleteButton>;

export const TrashIcon: Story = {
  name: "Trash Icon (Default)",
  args: { onDelete: () => alert("Delete clicked") },
};

export const XIcon: Story = {
  args: {
    icon: "x",
    onDelete: () => alert("Delete clicked"),
  },
};

export const NoLabel: Story = {
  name: "Label",
  args: {
    label: "This is the label",
    onDelete: () => alert("Delete clicked"),
  },
};
