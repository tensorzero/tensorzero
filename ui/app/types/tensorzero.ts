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
