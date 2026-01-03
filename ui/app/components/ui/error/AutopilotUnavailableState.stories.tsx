import type { Meta, StoryObj } from "@storybook/react-vite";
import { AutopilotUnavailableState } from "./AutopilotUnavailableState";

const meta = {
  title: "Autopilot/AutopilotUnavailableState",
  component: AutopilotUnavailableState,
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof AutopilotUnavailableState>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
