import { useState } from "react";
import type { VariantInfo, ChatCompletionConfig } from "tensorzero-node";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet";
import { Button } from "~/components/ui/button";
import { Label } from "~/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Separator } from "~/components/ui/separator";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";

interface VariantEditorProps {
  variantInfo: VariantInfo;
  confirmVariantInfo: (info: VariantInfo) => void;
  isOpen: boolean;
  onClose: () => void;
  variantName?: string;
}

export function VariantEditor({
  variantInfo,
  confirmVariantInfo,
  isOpen,
  onClose,
  variantName,
}: VariantEditorProps) {
  const config =
    variantInfo.inner.type === "chat_completion" ? variantInfo.inner : null;
  const [editedConfig, setEditedConfig] = useState<ChatCompletionConfig | null>(
    config,
  );

  if (variantInfo.inner.type !== "chat_completion" || !config) {
    return (
      <Sheet open={isOpen} onOpenChange={onClose}>
        <SheetContent>
          <div className="flex h-full items-center justify-center">
            <p className="text-muted-foreground">Unsupported variant type</p>
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  const updateTemplate = (
    type: "system" | "user" | "assistant",
    contents: string,
  ) => {
    setEditedConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        [`${type}_template`]: contents
          ? { contents, path: prev[`${type}_template`]?.path || "" }
          : null,
      };
    });
  };

  const handleSave = () => {
    if (!editedConfig) return;
    confirmVariantInfo({
      inner: {
        ...editedConfig,
        type: "chat_completion",
      },
      timeouts: variantInfo.timeouts,
    });
    onClose();
  };

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent side="right" className="w-full md:w-5/6">
        <SheetHeader>
          <SheetTitle>
            Edit Variant Configuration
            {variantName && (
              <span className="text-muted-foreground ml-2 font-normal">
                ({variantName})
              </span>
            )}
          </SheetTitle>
          <SheetDescription>
            Configure chat completion settings for this variant
          </SheetDescription>
        </SheetHeader>

        <div className="mt-4 max-h-[calc(100vh-12rem)] space-y-6 overflow-y-auto">
          {/* Weight */}
          <Card>
            <CardHeader>
              <CardTitle>Weight</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label>Variant Weight</Label>
                <div className="text-sm">
                  {config.weight !== null ? (
                    <Badge variant="secondary">{config.weight}</Badge>
                  ) : (
                    <span className="text-muted-foreground">Not set</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Model</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label>Model</Label>
                <div className="flex items-center gap-2">
                  <code className="bg-muted rounded px-2 py-1 font-mono text-sm">
                    {config.model}
                  </code>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Templates Section */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Templates</h3>
            {/* TODO before editing: fix the case where if you clear a template the template is no longer editable */}

            <div className="space-y-2">
              <Label>System Template</Label>
              {editedConfig?.system_template ? (
                <CodeEditor
                  value={editedConfig.system_template.contents}
                  allowedLanguages={["jinja2", "text"]}
                  onChange={(value) => updateTemplate("system", value)}
                  className="min-h-[200px]"
                />
              ) : (
                <div className="rounded-md border border-dashed p-8 text-center">
                  <p className="text-muted-foreground text-sm">
                    No system template defined
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label>User Template</Label>
              {editedConfig?.user_template ? (
                <CodeEditor
                  value={editedConfig.user_template.contents}
                  allowedLanguages={["jinja2", "text"]}
                  onChange={(value) => updateTemplate("user", value)}
                  className="min-h-[200px]"
                />
              ) : (
                <div className="rounded-md border border-dashed p-8 text-center">
                  <p className="text-muted-foreground text-sm">
                    No user template defined
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label>Assistant Template</Label>
              {editedConfig?.assistant_template ? (
                <CodeEditor
                  value={editedConfig.assistant_template.contents}
                  allowedLanguages={["jinja2", "text"]}
                  onChange={(value) => updateTemplate("assistant", value)}
                  className="min-h-[200px]"
                />
              ) : (
                <div className="rounded-md border border-dashed p-8 text-center">
                  <p className="text-muted-foreground text-sm">
                    No assistant template defined
                  </p>
                </div>
              )}
            </div>
          </div>

          <Separator />

          {/* Response Format */}
          <Card>
            <CardHeader>
              <CardTitle>Response Format</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>JSON Mode</Label>
                <div className="text-sm">
                  {config.json_mode ? (
                    <Badge>{config.json_mode}</Badge>
                  ) : (
                    <span className="text-muted-foreground">Disabled</span>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Stop Sequences</Label>
                <div className="text-sm">
                  {config.stop_sequences && config.stop_sequences.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {config.stop_sequences.map((seq, idx) => (
                        <Badge
                          key={idx}
                          variant="secondary"
                          className="font-mono"
                        >
                          "{seq}"
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground">None</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Generation Parameters */}
          <Card>
            <CardHeader>
              <CardTitle>Generation Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Temperature</Label>
                  <div className="text-sm">
                    {config.temperature !== null ? (
                      <Badge variant="outline">{config.temperature}</Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Top P</Label>
                  <div className="text-sm">
                    {config.top_p !== null ? (
                      <Badge variant="outline">{config.top_p}</Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Max Tokens</Label>
                  <div className="text-sm">
                    {config.max_tokens !== null ? (
                      <Badge variant="outline">{config.max_tokens}</Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Seed</Label>
                  <div className="text-sm">
                    {config.seed !== null ? (
                      <Badge variant="outline">{config.seed}</Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Penalties */}
          <Card>
            <CardHeader>
              <CardTitle>Penalties</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Presence Penalty</Label>
                  <div className="text-sm">
                    {config.presence_penalty !== null ? (
                      <Badge variant="outline">{config.presence_penalty}</Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Frequency Penalty</Label>
                  <div className="text-sm">
                    {config.frequency_penalty !== null ? (
                      <Badge variant="outline">
                        {config.frequency_penalty}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Timeout Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Timeout Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Non-Streaming Total (ms)</Label>
                  <div className="text-sm">
                    {variantInfo.timeouts.non_streaming.total_ms !== null ? (
                      <Badge variant="outline">
                        {variantInfo.timeouts.non_streaming.total_ms.toString()}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">Not set</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Streaming TTFT (ms)</Label>
                  <div className="text-sm">
                    {variantInfo.timeouts.streaming.ttft_ms !== null ? (
                      <Badge variant="outline">
                        {variantInfo.timeouts.streaming.ttft_ms.toString()}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">Not set</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Retry Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Retry Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Num Retries</Label>
                  <Badge variant="outline">{config.retries.num_retries}</Badge>
                </div>
                <div className="space-y-2">
                  <Label>Max Delay (s)</Label>
                  <Badge variant="outline">{config.retries.max_delay_s}</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <SheetFooter className="mt-6">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Changes</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
