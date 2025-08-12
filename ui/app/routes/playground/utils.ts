import type { DisplayInput } from "~/utils/clickhouse/common";
import { z } from "zod";
import type {
  Datapoint as TensorZeroDatapoint,
  InferenceResponse,
  VariantInfo,
  ClientInferenceParams,
  FunctionConfig,
} from "tensorzero-node";
import { prepareInferenceActionRequest } from "../api/tensorzero/inference.utils";
import { getExtraInferenceOptions } from "~/utils/feature_flags";
import { data } from "react-router";

export function isEditedVariantName(variantName: string): boolean {
  return variantName.startsWith("tensorzero::edited::");
}

export function refreshClientInference(
  setPromise: (
    outerKey: string,
    innerKey: string,
    promise: Promise<InferenceResponse>,
  ) => void,
  input: DisplayInput,
  datapoint: TensorZeroDatapoint,
  variant: PlaygroundVariantInfo,
  functionName: string,
  functionConfig: FunctionConfig,
) {
  const request = preparePlaygroundInferenceRequest(
    variant,
    functionName,
    datapoint,
    input,
    functionConfig,
  );
  // The API endpoint takes form data so we need to stringify it and send as data
  const formData = new FormData();
  formData.append("data", JSON.stringify(request));
  const responsePromise = async () => {
    const response = await fetch("/api/tensorzero/inference", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data;
  };
  setPromise(variant.name, datapoint.id, responsePromise());
}

const BuiltInVariantInfoSchema = z.object({
  type: z.literal("builtin"),
  name: z.string(),
});

const EditedVariantInfoSchema = z.object({
  type: z.literal("edited"),
  name: z.string(),
  config: z.custom<VariantInfo>((val) => {
    // Basic validation that it's an object
    return typeof val === "object" && val !== null;
  }),
});

const PlaygroundVariantInfoSchema = z.discriminatedUnion("type", [
  BuiltInVariantInfoSchema,
  EditedVariantInfoSchema,
]);
export type PlaygroundVariantInfo = z.infer<typeof PlaygroundVariantInfoSchema>;

export const SelectedVariantsSchema = z.array(PlaygroundVariantInfoSchema);
type SelectedVariants = z.infer<typeof SelectedVariantsSchema>;

export function getVariants(searchParams: URLSearchParams): SelectedVariants {
  const variants = searchParams.get("variants") ?? "[]";
  let parsedVariants;
  try {
    parsedVariants = JSON.parse(variants);
  } catch (error) {
    throw data(`Variant parameter must be valid JSON: ${error}`, {
      status: 400,
    });
  }

  const result = SelectedVariantsSchema.safeParse(parsedVariants);

  if (!result.success) {
    const errorDetails = result.error.issues
      .map((issue) => `${issue.path.join(".")}: ${issue.message}`)
      .join("; ");
    throw data(`Invalid variants parameter: ${errorDetails}`, { status: 400 });
  }
  return result.data;
}

/*
  We use the convention that variants
  beginning with "tensorzero::edited::" are edited variants.
  If the current variant is already "tensorzero::edited::*" we can reuse the existing name.

  We add a random identifier so that each edit gives a new name.

  This function checks if this is needed and then returns the new variant name.
*/
export function getNewVariantName(currentVariantName: string): string {
  let originalVariantName = currentVariantName;
  if (isEditedVariantName(currentVariantName)) {
    originalVariantName =
      extractOriginalVariantNameFromEdited(currentVariantName);
  }
  // generate a random identifier here so that each edit is unique
  const randomId = Math.random().toString(36).substring(2, 15);
  return `tensorzero::edited::${randomId}::${originalVariantName}`;
}

export function extractOriginalVariantNameFromEdited(
  editedVariantName: string,
): string {
  const match = editedVariantName.match(/^tensorzero::edited::[^:]*::(.*)$/);
  if (!match) {
    throw Error("Malformed variant name");
  }
  return match[1];
}

interface variantInferenceInfo {
  variant: string | undefined;
  editedVariantInfo: VariantInfo | undefined;
}

function getVariantInferenceInfo(
  variantInfo: PlaygroundVariantInfo,
): variantInferenceInfo {
  switch (variantInfo.type) {
    case "builtin":
      return {
        variant: variantInfo.name,
        editedVariantInfo: undefined,
      };
    case "edited":
      return {
        variant: undefined,
        editedVariantInfo: variantInfo.config,
      };
  }
}

export function preparePlaygroundInferenceRequest(
  variantInfo: PlaygroundVariantInfo,
  functionName: string,
  datapoint: TensorZeroDatapoint,
  input: DisplayInput,
  functionConfig: FunctionConfig,
): ClientInferenceParams {
  const variantInferenceInfo = getVariantInferenceInfo(variantInfo);
  const request = prepareInferenceActionRequest({
    source: "clickhouse_datapoint",
    input,
    functionName,
    variant: variantInferenceInfo.variant,
    tool_params:
      datapoint?.type === "chat"
        ? (datapoint.tool_params ?? undefined)
        : undefined,
    output_schema: datapoint?.type === "json" ? datapoint.output_schema : null,
    // The default is write_only but we do off in the playground
    cache_options: {
      max_age_s: null,
      enabled: "off",
    },
    dryrun: true,
    editedVariantInfo: variantInferenceInfo.editedVariantInfo,
    functionConfig,
  });
  const extraOptions = getExtraInferenceOptions();
  return {
    ...request,
    ...extraOptions,
  };
}
