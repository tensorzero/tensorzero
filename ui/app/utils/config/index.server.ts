import { getNativeTensorZeroClient } from "../supervised_fine_tuning/client";
import type { Config } from "tensorzero-node";

const CACHE_TTL_MS = 1000 * 60; // 1 minute

/*
Config Context provider:

In general, the config tree for TensorZero is static and can be loaded at startup and then used by any component.
This is good so that we can avoid reading the config from the file system on every request.
Since it is required for a very large number of components, it is also great to avoid drilling it down through nearly all components.

So we implement a context provider that loads the config at the root of the app and makes it available to all components
via the ConfigProvider and the useConfig hook.

However, there is one exception to this static behavior: the default function `tensorzero::default`.
Since the default function can be called with any model and since doing so with essentially creates a new variant,
we must check what variants have been used in the past for this function.

In order to avoid drilling the config through the entire application, we implement a caching mechanism here that is used for context.
We only reload the config (file + database query) if the config is needed (via the hook or a backend helper function getConfig)
and it has not been loaded in the past CACHE_TTL_MS.

This introduces a small liveness issue where the list of variants for the default function is not updated for up toCACHE_TTL_MS
after a new variant is used.

We will likely address this with some form of query library down the line.
*/

export async function loadConfig(): Promise<Config> {
  const tensorZeroClient = await getNativeTensorZeroClient();
  const config = await tensorZeroClient.getConfig();
  return config;
}

interface ConfigCache {
  data: Config;
  timestamp: number;
}

let configCache: ConfigCache | null = null;

export async function getConfig() {
  const now = Date.now();

  if (configCache && now - configCache.timestamp < CACHE_TTL_MS) {
    return configCache.data;
  }

  // Cache is invalid or doesn't exist, reload it
  const freshConfig = await loadConfig();

  configCache = { data: freshConfig, timestamp: now };
  return freshConfig;
}
