/**
 * Re-export tensorzero-node types through a local module so UI code can import
 * them without forcing Vite/Storybook to bundle the native client.
 *
 * Because this file only uses `export type`, the generated JavaScript tree
 * contains no runtime import of `tensorzero-node`.
 *
 * When importing Rust binding types in browser components, prefer this module.
 *
 * BAD: `import type { StoredInput } from "tensorzero-node";`
 * GOOD: `import type { StoredInput } from "~/types/tensorzero";`
 */
export type * from "tensorzero-node";

/**
 * Response for the autopilot status endpoint.
 * Indicates whether autopilot is configured on the gateway.
 */
export interface AutopilotStatusResponse {
  enabled: boolean;
}

/**
 * Cost aggregation row from GET /internal/functions/{name}/cost_by_variant.
 * Backend: VariantCost (#6264).
 */
export interface VariantCost {
  period_start: string;
  variant_name: string;
  total_cost: number;
  inference_count: number;
  inferences_with_cost: number;
}

/**
 * Response for the cost by variant endpoint.
 */
export interface GetFunctionCostByVariantResponse {
  cost: VariantCost[];
}
