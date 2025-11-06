import { useRef, useState } from "react";
import type { VariantInfo, ChatCompletionConfig } from "~/types/tensorzero";
import { Sheet, SheetContent, SheetFooter } from "~/components/ui/sheet";
import { PageHeader } from "~/components/layout/PageLayout";
import { Button } from "~/components/ui/button";
import { Label } from "~/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { CodeEditor } from "~/components/ui/code-editor";

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

  // Which templates existed when we opened?
  const initialTemplateNames = useRef<string[]>(
    config ? Object.keys(config.templates) : [],
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

  const updateTemplate = (templateName: string, contents: string) => {
    setEditedConfig((prev) => {
      if (!prev) return prev;

      const hadInitially = initialTemplateNames.current.includes(templateName);
      if (hadInitially) {
        // Keep an object even when empty so the editor stays visible/editable.
        const prevTemplate = prev.templates[templateName];
        const templates = {
          ...prev.templates,
        };
        if (prevTemplate) {
          templates[templateName] = {
            ...prevTemplate,
            template: { contents, path: prevTemplate.template.path },
          };
        }
        return {
          ...prev,
          templates,
        };
      } else {
        // Don't allow adding a new template if it's not initially present.
        // (at least for now)
        return prev;
      }
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
        <PageHeader label="Variant Configuration" name={variantName} />

        <div className="mt-4 max-h-[calc(100vh-12rem)] space-y-6 overflow-y-auto">
          {/* Weight */}
          <Card>
            <CardHeader>
              <CardTitle>Basic</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-4">
                  <Label>Model</Label>
                  <div className="font-mono text-sm">{config.model}</div>
                </div>
                <div className="space-y-4">
                  <Label>Weight</Label>
                  <div className="text-sm">
                    {config.weight !== null ? (
                      <div className="font-mono text-sm">{config.weight}</div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Templates Section */}
          {initialTemplateNames.current.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Templates</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {initialTemplateNames.current.map((templateName) => {
                  return (
                    <div key={templateName} className="space-y-4">
                      <Label>{templateName}</Label>
                      <CodeEditor
                        value={
                          editedConfig?.templates[templateName]?.template
                            ?.contents ?? ""
                        }
                        allowedLanguages={["jinja2", "text"]}
                        onChange={(value) =>
                          updateTemplate(templateName, value)
                        }
                        className="min-h-[200px]"
                      />
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}

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
                    <div className="font-mono text-sm">{config.json_mode}</div>
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
                        <div key={idx} className="font-mono text-sm">
                          "{seq}"
                        </div>
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
                      <div className="font-mono text-sm">
                        {config.temperature}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Top P</Label>
                  <div className="text-sm">
                    {config.top_p !== null ? (
                      <div className="font-mono text-sm">{config.top_p}</div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Max Tokens</Label>
                  <div className="text-sm">
                    {config.max_tokens !== null ? (
                      <div className="font-mono text-sm">
                        {config.max_tokens}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Seed</Label>
                  <div className="text-sm">
                    {config.seed !== null ? (
                      <div className="font-mono text-sm">{config.seed}</div>
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
                      <div className="font-mono text-sm">
                        {config.presence_penalty}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Frequency Penalty</Label>
                  <div className="text-sm">
                    {config.frequency_penalty !== null ? (
                      <div className="font-mono text-sm">
                        {config.frequency_penalty}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Timeouts */}
          <Card>
            <CardHeader>
              <CardTitle>Timeouts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Non-Streaming Total (ms)</Label>
                  <div className="text-sm">
                    {variantInfo.timeouts.non_streaming.total_ms !== null ? (
                      <div className="font-mono text-sm">
                        {variantInfo.timeouts.non_streaming.total_ms.toString()}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Streaming TTFT (ms)</Label>
                  <div className="text-sm">
                    {variantInfo.timeouts.streaming.ttft_ms !== null ? (
                      <div className="font-mono text-sm">
                        {variantInfo.timeouts.streaming.ttft_ms.toString()}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">Default</span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Retry */}
          <Card>
            <CardHeader>
              <CardTitle>Retries</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Number of Retries</Label>
                  <div className="font-mono text-sm">
                    {config.retries.num_retries}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Max Delay (s)</Label>
                  <div className="font-mono text-sm">
                    {config.retries.max_delay_s}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <SheetFooter className="mt-6 items-center">
          <span className="text-muted-foreground mr-8 text-xs">
            Changes don't affect the gateway. They are temporary and live in the
            browser.
          </span>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Changes</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
