import { useState } from "react";
import { useFetcher } from "react-router";
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
import { useCopy } from "~/hooks/use-copy";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { Check, Copy } from "lucide-react";

interface GenerateApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ActionData {
  apiKey?: string;
  error?: string;
}

export function GenerateApiKeyModal({
  isOpen,
  onClose,
}: GenerateApiKeyModalProps) {
  const fetcher = useFetcher<ActionData>();
  const { copy, didCopy, isCopyAvailable } = useCopy();
  const isReadOnly = useReadOnly();
  const [description, setDescription] = useState("");

  const isSubmitting = fetcher.state === "submitting";
  const apiKey = fetcher.data?.apiKey;
  const error = fetcher.data?.error;

  const handleCopy = async () => {
    if (apiKey) {
      await copy(apiKey);
    }
  };

  return (
    <ReadOnlyGuard>
      <Dialog open={isOpen} onOpenChange={onClose}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>
              {apiKey ? "API Key Generated" : "Generate API Key"}
            </DialogTitle>
            <DialogDescription>
              {apiKey
                ? "Save this key securely. You won't be able to see it again."
                : "Create a new API key to authenticate with the TensorZero Gateway."}
            </DialogDescription>
          </DialogHeader>

          <DialogBody>
            {apiKey ? (
              // Success state - show the generated API key
              <div className="space-y-4 px-1">
                <div className="space-y-2">
                  <Label>Your API Key</Label>
                  <div className="relative">
                    <pre className="bg-muted text-foreground/50 overflow-x-auto rounded-md border p-3 font-mono text-sm">
                      {apiKey}
                    </pre>
                    {isCopyAvailable && (
                      <Button
                        type="button"
                        size="iconSm"
                        variant="secondary"
                        onClick={handleCopy}
                        className="absolute top-2 right-2 border"
                        title="Copy to clipboard"
                      >
                        {didCopy ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </Button>
                    )}
                  </div>
                </div>
                {description && (
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <p className="text-muted-foreground text-sm">
                      {description}
                    </p>
                  </div>
                )}
                <div className="rounded-md border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-800 dark:bg-yellow-950">
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">
                    Make sure to copy your API key now. For security reasons,
                    you won't be able to see it again.
                  </p>
                </div>
              </div>
            ) : (
              // Form state - collect optional description
              <fetcher.Form method="post" className="space-y-4 px-1">
                <input type="hidden" name="action" value="generate" />
                <div className="space-y-2">
                  <Label htmlFor="description">
                    Description{" "}
                    <span className="text-muted-foreground">(optional)</span>
                  </Label>
                  <Textarea
                    id="description"
                    name="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Example: Production API key for web app"
                    maxLength={255}
                    rows={3}
                    className="resize-none focus-visible:ring-offset-0"
                  />
                  <p className="text-muted-foreground text-xs">
                    {description.length}/255 characters
                  </p>
                </div>
                {error && (
                  <div className="rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-950">
                    <p className="text-sm text-red-800 dark:text-red-200">
                      {error}
                    </p>
                  </div>
                )}
              </fetcher.Form>
            )}
          </DialogBody>

          <DialogFooter>
            {apiKey ? (
              <Button type="button" variant="outline" onClick={onClose}>
                Close
              </Button>
            ) : (
              <>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onClose}
                  disabled={isSubmitting}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={isReadOnly || isSubmitting}
                  onClick={() => {
                    const formData = new FormData();
                    formData.append("action", "generate");
                    if (description) {
                      formData.append("description", description);
                    }
                    fetcher.submit(formData, { method: "post" });
                  }}
                >
                  {isSubmitting ? "Generating..." : "Generate Key"}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ReadOnlyGuard>
  );
}
