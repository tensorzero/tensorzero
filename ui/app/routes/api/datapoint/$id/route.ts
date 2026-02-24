import { data, type LoaderFunctionArgs } from "react-router";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { loadFileDataForInput } from "~/utils/resolve.server";
import type { Datapoint, Input } from "~/types/tensorzero";

export interface DatapointDetailData {
  datapoint: Datapoint;
  resolvedInput: Input;
  dataset_name: string;
  function_name: string;
}

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { id } = params;

  if (!id) {
    throw data("Datapoint ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();

    const resolved = await client.resolveUuid(id);
    const datapointObj = resolved.object_types.find(
      (obj) => obj.type === "chat_datapoint" || obj.type === "json_datapoint",
    );

    if (!datapointObj) {
      throw data(`No datapoint found for id ${id}.`, { status: 404 });
    }

    if (
      datapointObj.type !== "chat_datapoint" &&
      datapointObj.type !== "json_datapoint"
    ) {
      throw data(`UUID ${id} does not resolve to a datapoint.`, {
        status: 404,
      });
    }

    const { dataset_name, function_name } = datapointObj;

    const datapoint = await client.getDatapoint(id, dataset_name);

    if (!datapoint) {
      throw data(`No datapoint found for id ${id}.`, { status: 404 });
    }

    const resolvedInput = await loadFileDataForInput(datapoint.input);

    const result: DatapointDetailData = {
      datapoint,
      resolvedInput,
      dataset_name,
      function_name,
    };

    return Response.json(result);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch datapoint:", error);
    throw data("Failed to fetch datapoint details", { status: 500 });
  }
}
