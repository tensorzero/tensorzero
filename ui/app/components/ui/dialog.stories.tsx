import {
  Dialog,
  DialogBody,
  DialogHeader,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from "./dialog";
import { Button } from "./button";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useArgs } from "storybook/preview-api";

const meta = {
  title: "DS/Dialog",
  component: Dialog,
} satisfies Meta<typeof Dialog>;

export default meta;
type Story = StoryObj<typeof meta>;

export const BasicDialog: Story = {
  args: {
    open: true,
  },
  render: function BasicDialog(args) {
    const [{ open }, updateArgs] = useArgs<{ open: boolean }>();
    return (
      <Dialog
        {...args}
        open={open}
        onOpenChange={(open) => updateArgs({ open })}
      >
        <DialogContent className="max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>Dialog title</DialogTitle>
            <DialogDescription>
              This is a description of the dialog. It can be used to provide
              additional information or instructions to the user.
            </DialogDescription>
          </DialogHeader>
          <DialogBody>
            <p>
              This is the body of the dialog. It can contain any content you
              want, including forms, text, images, etc.
            </p>
            <p>
              If the dialog content causes content to overflow, this section
              will be scrollable, while the header and footer will remain
              visible on either side.
            </p>
          </DialogBody>
          <DialogFooter>
            <Button onClick={() => updateArgs({ open: false })}>Ok</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  },
};

export const WithLongContent: Story = {
  args: {
    open: true,
  },
  render: function BasicDialog(args) {
    const [{ open }, updateArgs] = useArgs<{ open: boolean }>();
    return (
      <Dialog
        {...args}
        open={open}
        onOpenChange={(open) => updateArgs({ open })}
      >
        <DialogContent className="max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>Dialog title</DialogTitle>
            <DialogDescription>
              This is a description of the dialog. It can be used to provide
              additional information or instructions to the user.
            </DialogDescription>
          </DialogHeader>
          <DialogBody>
            <div className="flex flex-col gap-4 whitespace-pre-wrap break-words rounded-md bg-gray-100 p-4 font-mono text-xs">
              {Array.from({ length: 16 }).map((_, i) => (
                <p key={i}>
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
                  do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  Ut enim ad minim veniam, quis nostrud exercitation ullamco
                  laboris nisi ut aliquip ex ea commodo consequat. Duis aute
                  irure dolor in reprehenderit in voluptate velit esse cillum
                  dolore eu fugiat nulla pariatur. Excepteur sint occaecat
                  cupidatat non proident, sunt in culpa qui officia deserunt
                  mollit anim id est laborum.
                </p>
              ))}
            </div>
          </DialogBody>
          <DialogFooter>
            <Button onClick={() => updateArgs({ open: false })}>Ok</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  },
};
