import { useEffect, useState } from "react";
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
import {
  computeExpiresAt,
  EXPIRATION_PRESET_OPTIONS,
  getCustomExpirationError,
  isExpirationPreset,
  type ExpirationPreset,
} from "./expiration";

interface GenerateApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ActionData {
  apiKey?: string;
  error?: string;
  fieldErrors?: {
    expiresAt?: string;
  };
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
  const [submittedExpiresAt, setSubmittedExpiresAt] = useState<
    Date | undefined
  >(undefined);
  const [customDateEditedSinceSubmit, setCustomDateEditedSinceSubmit] =
    useState(false);

  useEffect(() => {
    if (isOpen) {
      setDescription("");
      setExpirationPreset("none");
      setCustomExpiresAt(undefined);
      setSubmittedExpiresAt(undefined);
      setCustomDateEditedSinceSubmit(false);
    }
  }, [isOpen]);

  const isSubmitting = fetcher.state === "submitting";
  const apiKey = fetcher.data?.apiKey;
  const error = fetcher.data?.error;
  const isCustomExpiresAtMissing =
    expirationPreset === "custom" && customExpiresAt === undefined;
  const customExpiresAtError =
    expirationPreset === "custom"
      ? (getCustomExpirationError(customExpiresAt) ??
        (!customDateEditedSinceSubmit
          ? (fetcher.data?.fieldErrors?.expiresAt ?? null)
          : null))
      : null;
  const hasCustomExpiresAtError = customExpiresAtError !== null;

  const handleCopy = async () => {
    if (apiKey) {
      await copy(apiKey);
    }
  };

  const handlePresetChange = (value: string) => {
    if (!isExpirationPreset(value)) {
      return;
    }

    setExpirationPreset(value);
  };

  const handleCustomExpiresAtChange = (nextExpiresAt: Date | undefined) => {
    setCustomExpiresAt(nextExpiresAt);
    setCustomDateEditedSinceSubmit(true);
  };

  const handleSubmit = () => {
    if (isCustomExpiresAtMissing || customExpiresAtError) {
      return;
    }

    const submittingExpiresAt =
      expirationPreset === "custom"
        ? customExpiresAt
        : computeExpiresAt(expirationPreset);

    const formData = new FormData();
    formData.append("action", "generate");
    formData.append("expiration_preset", expirationPreset);
    if (description) {
      formData.append("description", description);
    }
    if (submittingExpiresAt) {
      formData.append("expires_at", submittingExpiresAt.toISOString());
    }
    setSubmittedExpiresAt(submittingExpiresAt);
    setCustomDateEditedSinceSubmit(false);
    fetcher.submit(formData, { method: "post" });
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
              {submittedExpiresAt && (
                <div className="space-y-2">
                  <Label>Expires</Label>
                  <p className="text-muted-foreground text-sm">
                    {formatDate(submittedExpiresAt)}
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
                    {EXPIRATION_PRESET_OPTIONS.map(({ value, label }) => (
                      <SelectItem key={value} value={value}>
                        {label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {expirationPreset === "custom" && (
                  <div className="space-y-2">
                    <DateTimePicker
                      id="expires_at"
                      value={customExpiresAt}
                      onChange={handleCustomExpiresAtChange}
                      minDate={new Date()}
                      placeholder="Pick a date and time"
                      disabled={isSubmitting}
                      aria-invalid={hasCustomExpiresAtError}
                      aria-describedby={
                        hasCustomExpiresAtError ? "expires_at-error" : undefined
                      }
                    />
                    {hasCustomExpiresAtError && (
                      <p
                        id="expires_at-error"
                        className="text-destructive text-xs font-medium"
                      >
                        {customExpiresAtError}
                      </p>
                    )}
                  </div>
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
                type="button"
                disabled={
                  isReadOnly ||
                  isSubmitting ||
                  isCustomExpiresAtMissing ||
                  customExpiresAtError !== null
                }
                onClick={handleSubmit}
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
