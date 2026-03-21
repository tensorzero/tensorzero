# Model Providers

- We are gradually migrating types related to model providers to a separate crate `tensorzero-types-providers`. If you're creating a new type, (e.g. `MyProviderUsage`), prefer creating it under `tensorzero-types-providers` and importing it.
- For **prompt caching support**, see `tensorzero-types-providers/src/cache.rs`. It is the single source of truth for which providers support caching, their API field names, and how they map to TensorZero's internal `Usage` struct. Follow the checklist there when adding or modifying cache token support for a provider.
