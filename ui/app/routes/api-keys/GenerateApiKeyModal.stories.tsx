import type { Meta, StoryObj } from "@storybook/react-vite";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Label } from "~/components/ui/label";
import { Textarea } from "~/components/ui/textarea";
import { useArgs } from "storybook/preview-api";
import { Copy } from "lucide-react";

const meta = {
  title: "API Keys/GenerateApiKeyModal",
  component: Dialog,
} satisfies Meta<typeof Dialog>;

export default meta;
type Story = StoryObj<typeof meta>;

export const FormView: Story = {
  args: {
    open: true,
  },
  render: function FormView(args) {
    const [{ open }, updateArgs] = useArgs<{ open: boolean }>();
    return (
      <Dialog
        {...args}
        open={open}
        onOpenChange={(open) => updateArgs({ open })}
      >
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Generate API Key</DialogTitle>
            <DialogDescription>
              Create a new API key to authenticate with the TensorZero Gateway.
            </DialogDescription>
          </DialogHeader>

          <DialogBody className="overflow-hidden px-1">
            <div className="space-y-4 px-1">
              <div className="space-y-2">
                <Label htmlFor="description">
                  Description{" "}
                  <span className="text-muted-foreground">(optional)</span>
                </Label>
                <Textarea
                  id="description"
                  placeholder="e.g., Production API key for mobile app"
                  maxLength={255}
                  rows={3}
                  className="resize-none focus-visible:ring-offset-0"
                />
                <p className="text-muted-foreground text-xs">
                  0/255 characters
                </p>
              </div>
            </div>
          </DialogBody>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => updateArgs({ open: false })}
            >
              Cancel
            </Button>
            <Button type="submit">Generate Key</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  },
};

export const SuccessView: Story = {
  args: {
    open: true,
  },
  render: function SuccessView(args) {
    const [{ open }, updateArgs] = useArgs<{ open: boolean }>();
    const apiKey =
      "sk-tz-a1b2c3d4e5f6-9876543210abcdefghijklmnopqrstuvwxyz1234567890";
    const description = "Production API key for mobile app";

    return (
      <Dialog
        {...args}
        open={open}
        onOpenChange={(open) => updateArgs({ open })}
      >
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>API Key Generated</DialogTitle>
            <DialogDescription>
              Save this key securely. You won't be able to see it again.
            </DialogDescription>
          </DialogHeader>

          <DialogBody className="overflow-hidden px-1">
            <div className="space-y-4 px-1">
              <div className="space-y-2">
                <Label>Your API Key</Label>
                <div className="relative">
                  <pre className="bg-muted text-foreground overflow-x-auto rounded-md border p-3 font-mono text-sm">
                    {apiKey}
                  </pre>
                  <Button
                    type="button"
                    size="iconSm"
                    variant="ghost"
                    className="absolute top-2 right-2"
                    title="Copy to clipboard"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <p className="text-muted-foreground text-sm">{description}</p>
              </div>
              <div className="rounded-md border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-800 dark:bg-yellow-950">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Warning:</strong> Make sure to copy your API key now.
                  For security reasons, you won't be able to see it again.
                </p>
              </div>
            </div>
          </DialogBody>

          <DialogFooter>
            <Button type="button" onClick={() => updateArgs({ open: false })}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  },
};
