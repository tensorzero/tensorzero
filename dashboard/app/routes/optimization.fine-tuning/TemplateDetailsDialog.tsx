import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Textarea } from "~/components/ui/textarea";
import { ChatCompletionConfig } from "~/utils/config/variant";

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
        <DialogContent className="sm:max-w-[625px] p-0 overflow-hidden">
          <div className="max-h-[90vh] overflow-y-auto p-6 rounded-lg">
            <DialogHeader>
              <DialogTitle>Template Details</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="space-y-4">
                {variant && (
                  <>
                    <div className="space-y-2">
                      <h4 className="font-medium leading-none">
                        System Template
                      </h4>
                      {chatCompletionVariants[variant]?.system_template ? (
                        <Textarea
                          readOnly
                          value={
                            chatCompletionVariants[variant]?.system_template
                          }
                          className="h-[200px] resize-none"
                        />
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          No system template.
                        </p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-medium leading-none">
                        User Template
                      </h4>
                      {chatCompletionVariants[variant]?.user_template ? (
                        <Textarea
                          readOnly
                          value={chatCompletionVariants[variant]?.user_template}
                          className="h-[200px] resize-none"
                        />
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          No user template.
                        </p>
                      )}
                    </div>
                  </>
                )}

                <div className="space-y-2">
                  <h4 className="font-medium leading-none">
                    Assistant Template
                  </h4>
                  {chatCompletionVariants[variant]?.assistant_template ? (
                    <Textarea
                      readOnly
                      value={
                        chatCompletionVariants[variant]?.assistant_template
                      }
                      className="h-[200px] resize-none"
                    />
                  ) : (
                    <p className="text-sm text-muted-foreground">
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
