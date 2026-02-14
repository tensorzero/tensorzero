import type { Route } from "./+types/resolve_uuid.route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import type { ResolveUuidResponse } from "~/types/tensorzero";

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
