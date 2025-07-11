import React from "react";
import type { VariantConfig } from "tensorzero-node";

// Consistent grid layout for all metadata displays
export const METADATA_GRID =
  "grid grid-cols-[max-content_max-content_1fr] items-center gap-x-3 gap-y-1";

// Format numbers consistently with monospace font
export function formatNumber(
  value: number | null | undefined,
): React.ReactNode {
  if (value === null || value === undefined) return null;
  return <span className="font-mono text-xs">{value}</span>;
}

// Format arrays with truncation
export function formatArray(arr: string[]): React.ReactNode {
  return <span className="truncate text-xs">{arr.join(", ")}</span>;
}

// Format JSON mode consistently
export function formatJsonMode(mode: string | null | undefined): string {
  if (!mode || mode === "off") return "off";
  return mode;
}

// Reusable metadata row component
interface MetadataRowProps {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  valueClassName?: string;
}

export function MetadataRow({
  icon,
  label,
  value,
  valueClassName = "",
}: MetadataRowProps) {
  return (
    <>
      {icon}
      <span className="text-muted-foreground text-xs">{label}</span>
      <span
        className={`text-foreground truncate text-xs ${valueClassName}`.trim()}
      >
        {value}
      </span>
    </>
  );
}

// Section header component
interface MetadataSectionProps {
  title: string;
  className?: string;
}

export function MetadataSection({
  title,
  className = "",
}: MetadataSectionProps) {
  return (
    <span
      className={`text-muted-foreground col-span-3 mt-3 text-xs font-semibold first:mt-0 ${className}`.trim()}
    >
      {title}
    </span>
  );
}

// Type guards and helpers for variant configs
export function hasEvaluator(
  variant: VariantConfig,
): variant is Extract<VariantConfig, { type: "best_of_n_sampling" }> {
  return variant.type === "best_of_n_sampling";
}

export function hasFuser(
  variant: VariantConfig,
): variant is Extract<VariantConfig, { type: "mixture_of_n" }> {
  return variant.type === "mixture_of_n";
}

export function hasDirectModel(
  variant: VariantConfig,
): variant is Extract<
  VariantConfig,
  { type: "chat_completion" | "chain_of_thought" | "dicl" }
> {
  return (
    variant.type === "chat_completion" ||
    variant.type === "chain_of_thought" ||
    variant.type === "dicl"
  );
}

// Get the relevant config object for a variant
export function getVariantConfig(variant: VariantConfig) {
  if (hasEvaluator(variant)) return variant.evaluator;
  if (hasFuser(variant)) return variant.fuser;
  return variant;
}
