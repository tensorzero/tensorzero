import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
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
                {variant && chatCompletionVariants[variant] ? (
                  Object.keys(chatCompletionVariants[variant].templates)
                    .length > 0 ? (
                    Object.entries(
                      chatCompletionVariants[variant].templates,
                    ).map(([templateName, templateData]) => {
                      const isLegacy = templateData?.legacy_definition === true;
                      return (
                        <div key={templateName} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <h4 className="font-mono leading-none font-medium">
                              {templateName}
                            </h4>
                            {isLegacy && (
                              <TooltipProvider delayDuration={200}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Badge className="bg-yellow-600 px-1 py-0 text-[10px] text-white">
                                      Legacy
                                    </Badge>
                                  </TooltipTrigger>
                                  <TooltipContent
                                    side="top"
                                    className="max-w-xs p-2"
                                  >
                                    <div className="text-xs">
                                      Please migrate from{" "}
                                      <code>{templateName}_template</code> to{" "}
                                      <code>templates.{templateName}.path</code>
                                      .{" "}
                                      <a
                                        href="https://www.tensorzero.com/docs/gateway/create-a-prompt-template#migrate-from-legacy-prompt-templates"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="underline hover:text-gray-300"
                                      >
                                        Read more
                                      </a>
                                    </div>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            )}
                          </div>
                          {templateData?.template ? (
                            <CodeEditor
                              value={templateData.template.contents}
                              readOnly
                            />
                          ) : (
                            <p className="text-muted-foreground text-sm">
                              No template defined.
                            </p>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      No templates defined.
                    </p>
                  )
                ) : null}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
