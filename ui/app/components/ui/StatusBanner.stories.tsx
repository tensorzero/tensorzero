import type { Meta, StoryObj } from "@storybook/react-vite";
import { AlertTriangle, Info, RefreshCw } from "lucide-react";
import { StatusBanner, StatusBannerVariant } from "./StatusBanner";

const meta: Meta<typeof StatusBanner> = {
  title: "StatusBanner",
  component: StatusBanner,
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof StatusBanner>;

export const Warning: Story = {
  args: {
    variant: StatusBannerVariant.Warning,
    children: "Failed to fetch events. Retrying...",
  },
};

export const WarningWithIcon: Story = {
  args: {
    variant: StatusBannerVariant.Warning,
    icon: RefreshCw,
    children: "Connection lost. Attempting to reconnect...",
  },
};

export const Error: Story = {
  args: {
    variant: StatusBannerVariant.Error,
    children:
      "Auto-approval failed for 2 tool calls. Retrying every 60 seconds...",
  },
};

export const ErrorWithIcon: Story = {
  args: {
    variant: StatusBannerVariant.Error,
    icon: AlertTriangle,
    children:
      "Auto-approval failed for 1 tool call. Retrying every 60 seconds...",
  },
};

export const WithCustomClassName: Story = {
  args: {
    variant: StatusBannerVariant.Warning,
    icon: Info,
    children: "This banner has custom margin.",
    className: "mt-8",
  },
};

export const LongContent: Story = {
  args: {
    variant: StatusBannerVariant.Error,
    icon: AlertTriangle,
    children:
      "This is a longer error message that demonstrates how the banner handles multiple lines of content when the text is too long to fit on a single line.",
  },
};
