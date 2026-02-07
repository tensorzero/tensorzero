import type { Meta, StoryObj } from "@storybook/react-vite";
import { AlertTriangle, Check } from "lucide-react";
import { Toaster } from "./toaster";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
import { useToast } from "~/hooks/use-toast";
import { Button } from "./button";

function ToastDemo({ variant }: { variant: "info" | "success" | "error" }) {
  const { toast } = useToast();

  const trigger = () => {
    if (variant === "error") {
      toast.error({
        title: "Something went wrong",
        description: "Could not complete the request. Please try again.",
      });
    } else if (variant === "success") {
      toast.success({
        title: "Datapoint added",
        description: "Successfully added to dataset.",
      });
    } else {
      toast.info({
        title: "Copied to clipboard",
        description: "The inference ID has been copied.",
      });
    }
  };

  return (
    <div>
      <Button onClick={trigger}>Show Toast</Button>
      <Toaster />
    </div>
  );
}

function ToastIconDemo({
  icon,
  iconClassName,
  title,
  description,
}: {
  icon?: React.ComponentType<{ className?: string }>;
  iconClassName?: string;
  title: string;
  description: string;
}) {
  const { toast } = useToast();

  return (
    <div>
      <Button
        onClick={() =>
          toast.info({ icon, iconClassName, title, description } as Parameters<
            typeof toast.info
          >[0])
        }
      >
        Show Toast
      </Button>
      <Toaster />
    </div>
  );
}

const meta = {
  title: "DS/Toast",
  decorators: [
    (Story) => (
      <GlobalToastProvider>
        <Story />
      </GlobalToastProvider>
    ),
  ],
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

export const Info: Story = {
  render: () => <ToastDemo variant="info" />,
};

export const Success: Story = {
  render: () => <ToastDemo variant="success" />,
};

export const Error: Story = {
  render: () => <ToastDemo variant="error" />,
};

export const CustomIcon: Story = {
  render: () => (
    <ToastIconDemo
      icon={AlertTriangle}
      iconClassName="text-amber-500 dark:text-amber-400"
      title="Rate limit warning"
      description="You are approaching the API rate limit."
    />
  ),
};

export const WithCheckIcon: Story = {
  render: () => (
    <ToastIconDemo
      icon={Check}
      iconClassName="text-green-500 dark:text-green-400"
      title="Changes saved"
      description="Your configuration has been updated."
    />
  ),
};
