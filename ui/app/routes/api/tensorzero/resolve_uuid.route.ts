import type { Route } from "./+types/resolve_uuid.route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import type { ResolveUuidResponse } from "~/types/tensorzero";

/**
 * Loader that resolves a UUID to determine what type of object it represents.
 * Returns the resolution result, or an empty object_types array on error.
 */
export async function loader({
  params,
}: Route.LoaderArgs): Promise<ResolveUuidResponse> {
  const id = params.uuid;
  if (!id) {
    return { id: "", object_types: [] };
  }

  try {
    return await getTensorZeroClient().resolveUuid(id);
  } catch (error) {
    logger.error("Failed to resolve UUID:", error);
    return { id, object_types: [] };
  }
}
