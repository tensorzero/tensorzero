import React from "react";
import {
  Thermometer,
  Hash,
  Zap,
  InfinityIcon,
  StopCircle,
  FileJson,
  RefreshCw,
  Settings,
  AlertCircle,
  Sparkles,
} from "lucide-react";
import {
  formatNumber,
  formatArray,
  formatJsonMode,
  MetadataRow,
  MetadataSection,
} from "./VariantMetadataUtils";

import type { ExtraHeadersConfig, ExtraBodyConfig } from "tensorzero-node";

interface BaseConfig {
  temperature?: number | null;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  stop_sequences?: string[] | null;
  json_mode?: string | null;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  retries: {
    num_retries: number;
    max_delay_s: number;
  };
  extra_headers?: ExtraHeadersConfig | null;
  extra_body?: ExtraBodyConfig | null;
}

// Sampling Section Component
export function SamplingSection({ config }: { config: BaseConfig }) {
  const hasSampling =
    config.temperature !== null ||
    config.top_p !== null ||
    config.seed !== null;

  if (!hasSampling) return null;

  return (
    <>
      <MetadataSection title="Sampling" />
      {config.temperature !== null && config.temperature !== undefined && (
        <MetadataRow
          icon={<Thermometer className="text-muted-foreground h-3.5 w-3.5" />}
          label="Temperature"
          value={formatNumber(config.temperature)}
        />
      )}
      {config.top_p !== null && config.top_p !== undefined && (
        <MetadataRow
          icon={<Sparkles className="text-muted-foreground h-3.5 w-3.5" />}
          label="Top P"
          value={formatNumber(config.top_p)}
        />
      )}
      {config.seed !== null && config.seed !== undefined && (
        <MetadataRow
          icon={<Hash className="text-muted-foreground h-3.5 w-3.5" />}
          label="Seed"
          value={formatNumber(config.seed)}
        />
      )}
    </>
  );
}

// Generation Section Component
export function GenerationSection({ config }: { config: BaseConfig }) {
  const hasGeneration =
    config.max_tokens !== null || config.stop_sequences || config.json_mode;

  if (!hasGeneration) return null;

  return (
    <>
      <MetadataSection title="Generation" />
      {config.max_tokens !== null && config.max_tokens !== undefined && (
        <MetadataRow
          icon={<InfinityIcon className="text-muted-foreground h-3.5 w-3.5" />}
          label="Max tokens"
          value={formatNumber(config.max_tokens)}
        />
      )}
      {config.stop_sequences && config.stop_sequences.length > 0 && (
        <MetadataRow
          icon={<StopCircle className="text-muted-foreground h-3.5 w-3.5" />}
          label="Stop sequences"
          value={formatArray(config.stop_sequences)}
        />
      )}
      {config.json_mode && (
        <MetadataRow
          icon={<FileJson className="text-muted-foreground h-3.5 w-3.5" />}
          label="JSON mode"
          value={formatJsonMode(config.json_mode)}
          valueClassName="font-mono"
        />
      )}
    </>
  );
}

// Penalties Section Component
export function PenaltiesSection({ config }: { config: BaseConfig }) {
  const hasPenalties =
    config.frequency_penalty !== null || config.presence_penalty !== null;

  if (!hasPenalties) return null;

  return (
    <>
      <MetadataSection title="Penalties" />
      {config.frequency_penalty !== null &&
        config.frequency_penalty !== undefined && (
          <MetadataRow
            icon={<Zap className="text-muted-foreground h-3.5 w-3.5" />}
            label="Frequency"
            value={formatNumber(config.frequency_penalty)}
          />
        )}
      {config.presence_penalty !== null &&
        config.presence_penalty !== undefined && (
          <MetadataRow
            icon={<AlertCircle className="text-muted-foreground h-3.5 w-3.5" />}
            label="Presence"
            value={formatNumber(config.presence_penalty)}
          />
        )}
    </>
  );
}

// System Section Component
export function SystemSection({ config }: { config: BaseConfig }) {
  return (
    <>
      <MetadataSection title="System" />
      <MetadataRow
        icon={<RefreshCw className="text-muted-foreground h-3.5 w-3.5" />}
        label="Max retries"
        value={formatNumber(config.retries.num_retries)}
      />
      <MetadataRow
        icon={<RefreshCw className="text-muted-foreground h-3.5 w-3.5" />}
        label="Max delay"
        value={<>{formatNumber(config.retries.max_delay_s)}s</>}
      />

      {config.extra_headers && (
        <MetadataRow
          icon={<Settings className="text-muted-foreground h-3.5 w-3.5" />}
          label="Extra headers"
          value={<>{formatNumber(config.extra_headers.data.length)} headers</>}
        />
      )}

      {config.extra_body && (
        <MetadataRow
          icon={<Settings className="text-muted-foreground h-3.5 w-3.5" />}
          label="Extra body"
          value={
            <>{formatNumber(config.extra_body.data.length)} replacements</>
          }
        />
      )}
    </>
  );
}
