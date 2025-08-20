import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Textarea } from "~/components/ui/textarea";
import type { ChatCompletionConfig } from "tensorzero-node";

interface TemplateDetailsDialogProps {
  variant: string;
  disabled: boolean;
  chatCompletionVariants: Record<string, ChatCompletionConfig>;
}

export function TemplateDetailsDialog({
  variant,
  disabled,
  chatCompletionVariants,
}: TemplateDetailsDialogProps) {
  return (
    <div className="flex">
      <Dialog>
        <DialogTrigger asChild>
          <Button variant="outline" disabled={disabled}>
            Details
          </Button>
        </DialogTrigger>
        <DialogContent className="overflow-hidden p-0 sm:max-w-[625px]">
          <div className="max-h-[90vh] overflow-y-auto rounded-lg p-6">
            <DialogHeader>
              <DialogTitle>Template Details</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="space-y-4">
                {variant && (
                  <>
                    <div className="space-y-2">
                      <h4 className="leading-none font-medium">
                        System Template
                      </h4>
                      {chatCompletionVariants[variant]?.templates.system
                        ?.template ? (
                        <Textarea
                          readOnly
                          value={
                            chatCompletionVariants[variant]?.templates.system
                              ?.template?.contents
                          }
                          className="h-[200px] resize-none"
                        />
                      ) : (
                        <p className="text-muted-foreground text-sm">
                          No system template.
                        </p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <h4 className="leading-none font-medium">
                        User Template
                      </h4>
                      {chatCompletionVariants[variant]?.templates.user
                        ?.template ? (
                        <Textarea
                          readOnly
                          value={
                            chatCompletionVariants[variant]?.templates.user
                              ?.template?.contents
                          }
                          className="h-[200px] resize-none"
                        />
                      ) : (
                        <p className="text-muted-foreground text-sm">
                          No user template.
                        </p>
                      )}
                    </div>
                  </>
                )}

                <div className="space-y-2">
                  <h4 className="leading-none font-medium">
                    Assistant Template
                  </h4>
                  {chatCompletionVariants[variant]?.templates.assistant
                    ?.template ? (
                    <Textarea
                      readOnly
                      value={
                        chatCompletionVariants[variant]?.templates.assistant
                          ?.template?.contents
                      }
                      className="h-[200px] resize-none"
                    />
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      No assistant template.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
