import { useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import { DateTimePicker } from "~/components/ui/date-time-picker";
import { formatDate } from "~/utils/date";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Textarea } from "~/components/ui/textarea";
import { useCopy } from "~/hooks/use-copy";
import { useReadOnly } from "~/context/read-only";
import { Check, Copy } from "lucide-react";

interface GenerateApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ActionData {
  apiKey?: string;
  error?: string;
}

type ExpirationPreset = "none" | "7d" | "30d" | "90d" | "1y" | "custom";

function computeExpiresAt(preset: ExpirationPreset): Date | undefined {
  const now = new Date();
  switch (preset) {
    case "7d":
      return new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
    case "30d":
      return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
    case "90d":
      return new Date(now.getTime() + 90 * 24 * 60 * 60 * 1000);
    case "1y":
      return new Date(now.getTime() + 365 * 24 * 60 * 60 * 1000);
    case "none":
    case "custom":
      return undefined;
  }
}

export function GenerateApiKeyModal({
  isOpen,
  onClose,
}: GenerateApiKeyModalProps) {
  const fetcher = useFetcher<ActionData>();
  const { copy, didCopy, isCopyAvailable } = useCopy();
  const isReadOnly = useReadOnly();
  const [description, setDescription] = useState("");
  const [expirationPreset, setExpirationPreset] =
    useState<ExpirationPreset>("none");
  const [customExpiresAt, setCustomExpiresAt] = useState<Date | undefined>(
    undefined,
  );

  const expiresAt =
    expirationPreset === "custom"
      ? customExpiresAt
      : computeExpiresAt(expirationPreset);

  const isSubmitting = fetcher.state === "submitting";
  const apiKey = fetcher.data?.apiKey;
  const error = fetcher.data?.error;

  const handleCopy = async () => {
    if (apiKey) {
      await copy(apiKey);
    }
  };

  const handlePresetChange = (value: string) => {
    setExpirationPreset(value as ExpirationPreset);
  };

  return (
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
                  <p className="text-muted-foreground text-sm">{description}</p>
                </div>
              )}
              {expiresAt && (
                <div className="space-y-2">
                  <Label>Expires</Label>
                  <p className="text-muted-foreground text-sm">
                    {formatDate(expiresAt)}
                  </p>
                </div>
              )}
              <div className="rounded-md border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-800 dark:bg-yellow-950">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  Make sure to copy your API key now. For security reasons, you
                  won't be able to see it again.
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
              <div className="space-y-2">
                <Label htmlFor="expiration-preset">
                  Expiration{" "}
                  <span className="text-muted-foreground">(optional)</span>
                </Label>
                <Select
                  value={expirationPreset}
                  onValueChange={handlePresetChange}
                >
                  <SelectTrigger id="expiration-preset">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">No expiration</SelectItem>
                    <SelectItem value="7d">7 days</SelectItem>
                    <SelectItem value="30d">30 days</SelectItem>
                    <SelectItem value="90d">90 days</SelectItem>
                    <SelectItem value="1y">1 year</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
                {expirationPreset === "custom" && (
                  <DateTimePicker
                    id="expires_at"
                    value={customExpiresAt}
                    onChange={setCustomExpiresAt}
                    minDate={new Date()}
                    placeholder="Pick a date and time"
                  />
                )}
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
                  if (expiresAt) {
                    formData.append("expires_at", expiresAt.toISOString());
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
  );
}
