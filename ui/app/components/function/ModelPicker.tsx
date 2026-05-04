/**
 * Two-level dropdown for picking a `<provider>::<model>` shorthand.
 *
 * Backed by `GET /internal/providers/available`, which returns the catalog
 * of supported providers, a flag for whether the gateway has credentials
 * for each, and a short hand-curated list of common models per provider.
 *
 * UX:
 * - Provider dropdown lists credentialed providers first (with a check icon),
 *   uncredentialed ones below (slightly muted).
 * - Model dropdown is filtered by the selected provider, with "Custom..."
 *   at the bottom that flips the field into a free-text input.
 * - The component owns a hidden `<input name={name}>` so it can drop into
 *   any HTML form alongside other fields.
 *
 * Server validation still happens at function-create time: typing a model
 * name that doesn't exist for that provider will fail at inference, exactly
 * like before. The picker is a guide, not a constraint.
 */

import { useEffect, useMemo, useState } from "react";
import { Check } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Input } from "~/components/ui/input";

interface ProviderInfo {
  id: string;
  display_name: string;
  credential_present: boolean;
  common_models: string[];
}

interface ProvidersResponse {
  providers: ProviderInfo[];
}

interface ModelPickerProps {
  /** Hidden input name. The submitted value is `<provider>::<model>`. */
  name: string;
  /** Initial value (e.g. for edit forms). Plain string in the same format. */
  defaultValue?: string;
  /** Called whenever the composed value changes. */
  onChange?: (value: string) => void;
  required?: boolean;
}

const CUSTOM_PROVIDER = "__custom__";
const CUSTOM_MODEL = "__custom__";

/**
 * Splits an `openai::gpt-4o-mini` style string into `[provider, model]`.
 * Returns `[null, null]` if the string doesn't have the `::` separator.
 */
function splitShorthand(value: string): [string | null, string | null] {
  const idx = value.indexOf("::");
  if (idx < 0) return [null, null];
  return [value.slice(0, idx), value.slice(idx + 2)];
}

export function ModelPicker({
  name,
  defaultValue,
  onChange,
  required,
}: ModelPickerProps) {
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Provider selection — `null` until catalog loads, then either a known
  // id, the sentinel CUSTOM_PROVIDER, or a free-text fallback.
  const [provider, setProvider] = useState<string | null>(null);
  const [customProvider, setCustomProvider] = useState<string>("");

  // Model selection — initial value mirrors provider; empty until provider
  // is set, then a known model from the provider's list, or CUSTOM_MODEL,
  // or a free-text fallback when the user picked "Custom...".
  const [model, setModel] = useState<string>("");
  const [customModel, setCustomModel] = useState<string>("");

  // Seed from `defaultValue` if provided. Done in an effect so it picks
  // up after the catalog arrives.
  useEffect(() => {
    if (!defaultValue || !providers) return;
    const [p, m] = splitShorthand(defaultValue);
    if (p && providers.some((entry) => entry.id === p)) {
      setProvider(p);
    } else if (p) {
      setProvider(CUSTOM_PROVIDER);
      setCustomProvider(p);
    }
    if (m) {
      setModel(m);
    }
  }, [defaultValue, providers]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/providers/available")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ProvidersResponse) => {
        if (cancelled) return;
        setProviders(data.providers);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const composedValue = useMemo(() => {
    const p = provider === CUSTOM_PROVIDER ? customProvider.trim() : provider;
    const m = model === CUSTOM_MODEL ? customModel.trim() : model;
    if (!p || !m) return "";
    return `${p}::${m}`;
  }, [provider, customProvider, model, customModel]);

  useEffect(() => {
    onChange?.(composedValue);
  }, [composedValue, onChange]);

  // Sort providers: credentialed first (and stable within each group)
  // so the user's most likely picks are always at the top.
  const sortedProviders = useMemo(() => {
    if (!providers) return [];
    const withCreds = providers.filter((p) => p.credential_present);
    const without = providers.filter((p) => !p.credential_present);
    return [...withCreds, ...without];
  }, [providers]);

  const selectedProviderInfo = useMemo(() => {
    if (!provider || provider === CUSTOM_PROVIDER) return null;
    return providers?.find((p) => p.id === provider) ?? null;
  }, [provider, providers]);

  if (error) {
    // Catalog fetch failed — degrade gracefully to free text. The user
    // can still type any provider::model string.
    return (
      <div className="flex flex-col gap-1.5">
        <Input
          name={name}
          defaultValue={defaultValue}
          placeholder="openai::gpt-4o-mini"
          required={required}
        />
        <span className="text-fg-muted text-xs">
          (Couldn&apos;t load provider catalog: {error}. Type the model string
          directly.)
        </span>
      </div>
    );
  }

  if (!providers) {
    return (
      <Input
        name={name}
        defaultValue={defaultValue}
        placeholder="loading providers…"
        required={required}
        disabled
      />
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Hidden field carries the composed value to form submission. */}
      <input
        type="hidden"
        name={name}
        value={composedValue}
        required={required}
      />

      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {/* Provider */}
        <Select
          value={provider ?? undefined}
          onValueChange={(v) => {
            setProvider(v);
            // Reset model selection whenever the provider changes —
            // models are scoped to a provider, so the previous choice
            // probably no longer makes sense.
            setModel("");
            setCustomModel("");
          }}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select a provider…" />
          </SelectTrigger>
          <SelectContent>
            {sortedProviders.map((p) => (
              <SelectItem key={p.id} value={p.id}>
                <span className="flex items-center gap-2">
                  {p.credential_present ? (
                    <Check className="h-3 w-3 text-emerald-600" />
                  ) : (
                    <span className="text-fg-muted h-3 w-3" />
                  )}
                  <span className={p.credential_present ? "" : "text-fg-muted"}>
                    {p.display_name}
                  </span>
                  {!p.credential_present && (
                    <span className="text-fg-muted text-xs">
                      (no credentials)
                    </span>
                  )}
                </span>
              </SelectItem>
            ))}
            <SelectItem value={CUSTOM_PROVIDER}>
              <span className="text-fg-secondary italic">Custom…</span>
            </SelectItem>
          </SelectContent>
        </Select>

        {/* Model */}
        {provider === CUSTOM_PROVIDER ? null : (
          <Select
            value={model || undefined}
            onValueChange={(v) => setModel(v)}
            disabled={!provider}
          >
            <SelectTrigger>
              <SelectValue
                placeholder={
                  provider ? "Select a model…" : "Pick a provider first"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {selectedProviderInfo?.common_models.map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
              <SelectItem value={CUSTOM_MODEL}>
                <span className="text-fg-secondary italic">Custom…</span>
              </SelectItem>
            </SelectContent>
          </Select>
        )}
      </div>

      {provider === CUSTOM_PROVIDER && (
        <Input
          value={customProvider}
          onChange={(e) => setCustomProvider(e.target.value)}
          placeholder="provider id (e.g. cohere)"
          aria-label="Custom provider id"
        />
      )}

      {(provider === CUSTOM_PROVIDER || model === CUSTOM_MODEL) && (
        <Input
          value={customModel}
          onChange={(e) => setCustomModel(e.target.value)}
          placeholder="model name (e.g. command-r-plus)"
          aria-label="Custom model name"
        />
      )}

      {composedValue && (
        <span className="text-fg-muted text-xs">
          Will be saved as <code className="text-xs">{composedValue}</code>
        </span>
      )}
    </div>
  );
}
