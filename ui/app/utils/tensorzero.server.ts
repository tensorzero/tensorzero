import { TensorZeroClient } from "./tensorzero";

if (!process.env.TENSORZERO_GATEWAY_URL) {
  throw new Error("TENSORZERO_GATEWAY_URL environment variable is required");
}

// Export a singleton instance
export const tensorZeroClient = new TensorZeroClient(
  process.env.TENSORZERO_GATEWAY_URL,
);
