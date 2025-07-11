import React from "react";
import type { VariantConfig } from "tensorzero-node";
import { METADATA_GRID, getVariantConfig } from "./VariantMetadataUtils";
import {
  SamplingSection,
  GenerationSection,
  PenaltiesSection,
  SystemSection,
} from "./VariantAdvancedSections";

interface VariantAdvancedMetadataProps {
  variant: VariantConfig;
}

export function VariantAdvancedMetadata({
  variant,
}: VariantAdvancedMetadataProps) {
  const config = getVariantConfig(variant);

  return (
    <div className={METADATA_GRID}>
      <SamplingSection config={config} />
      <GenerationSection config={config} />
      {(variant.type === "chat_completion" ||
        variant.type === "chain_of_thought" ||
        variant.type === "dicl") && <PenaltiesSection config={config} />}
      <SystemSection config={config} />
    </div>
  );
}
