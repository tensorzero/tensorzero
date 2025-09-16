import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import {
  data,
  isRouteErrorResponse,
  Link,
  redirect,
  useFetcher,
  useParams,
} from "react-router";
import { v7 as uuid } from "uuid";
import InputSnippet from "~/components/inference/InputSnippet";
import { Output } from "~/components/inference/Output";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { Badge } from "~/components/ui/badge";
import { TagsTable } from "~/components/tags/TagsTable";
import { useFunctionConfig } from "~/context/config";
import { resolvedInputToTensorZeroInput } from "~/routes/api/tensorzero/inference.utils";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";
import type { DisplayInputMessage } from "~/utils/clickhouse/common";

import {
  ParsedDatasetRowSchema,
  type ParsedDatasetRow,
} from "~/utils/clickhouse/datasets";
import {
  getDatapoint,
  getDatasetCounts,
  staleDatapoint,
} from "~/utils/clickhouse/datasets.server";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { Route } from "./+types/route";
import { DatapointActions } from "./DatapointActions";
import DatapointBasicInfo from "./DatapointBasicInfo";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
} from "tensorzero-node";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  try {
    const rawData = {
      dataset_name: formData.get("dataset_name"),
      function_name: formData.get("function_name"),
      id: formData.get("id"),
      episode_id: formData.get("episode_id"),
      input: JSON.parse(formData.get("input") as string),
      output: formData.get("output")
        ? JSON.parse(formData.get("output") as string)
        : undefined,
      output_schema: formData.get("output_schema")
        ? JSON.parse(formData.get("output_schema") as string)
        : undefined,
      tool_params: formData.get("tool_params")
        ? JSON.parse(formData.get("tool_params") as string)
        : undefined,
      tags: JSON.parse(formData.get("tags") as string),
      auxiliary: formData.get("auxiliary"),
      is_deleted: formData.get("is_deleted") === "true",
      updated_at: formData.get("updated_at"),
      staled_at: null,
      source_inference_id: formData.get("source_inference_id"),
      is_custom: true,
    };

    const cleanedData = Object.fromEntries(
      Object.entries(rawData).filter(([, value]) => value !== undefined),
    );
    const parsedFormData: ParsedDatasetRow =
      ParsedDatasetRowSchema.parse(cleanedData);
    const config = await getConfig();
    const functionConfig = await getFunctionConfig(
      parsedFormData.function_name,
      config,
    );
    if (!functionConfig) {
      return new Response(
        `Failed to find function config for function ${parsedFormData.function_name}`,
        { status: 400 },
      );
    }
    const functionType = functionConfig.type;

    const action = formData.get("action");
    if (action === "delete") {
      await staleDatapoint(
        parsedFormData.dataset_name,
        parsedFormData.id,
        functionType,
      );
      const datasetCounts = await getDatasetCounts({});
      const datasetCount = datasetCounts.find(
        (count) => count.dataset_name === parsedFormData.dataset_name,
      );

      if (datasetCount === undefined) {
        return redirect("/datasets");
      }
      return redirect(`/datasets/${parsedFormData.dataset_name}`);
    } else if (action === "save") {
      // If the input changed, we should remove the source_inference_id
      // because it will no longer be valid
      // Transform input to match TensorZero client's expected format
      const transformedInput = resolvedInputToTensorZeroInput(
        parsedFormData.input,
      );
      const transformedOutput = transformOutputForTensorZero(
        parsedFormData.output,
      );

      try {
        // For future reference:
        // These two calls would be a transaction but ClickHouse doesn't support

        const baseDatapoint = {
          function_name: parsedFormData.function_name,
          input: transformedInput,
          output: transformedOutput,
          tags: parsedFormData.tags || {},
          auxiliary: parsedFormData.auxiliary,
          is_custom: true, // we're saving it after an edit, so it's custom
          source_inference_id: parsedFormData.source_inference_id,
          id: uuid(), // We generate a new ID here because we want old evaluation runs to be able to point to the correct data.
        };

        let datapoint;
        if (functionType === "json" && "output_schema" in parsedFormData) {
          datapoint = {
            ...baseDatapoint,
            output_schema: parsedFormData.output_schema,
          };
        } else if (functionType === "chat" && "tool_params" in parsedFormData) {
          datapoint = {
            ...baseDatapoint,
            tool_params: parsedFormData.tool_params,
          };
        } else {
          throw new Error(
            `Unexpected function type "${functionType}" or missing required properties on datapoint`,
          );
        }
        const { id } = await getTensorZeroClient().updateDatapoint(
          parsedFormData.dataset_name,
          datapoint,
        );
        await staleDatapoint(
          parsedFormData.dataset_name,
          parsedFormData.id,
          functionType,
        );

        return redirect(
          `/datasets/${parsedFormData.dataset_name}/datapoint/${id}`,
        );
      } catch (error) {
        logger.error("Error updating datapoint:", error);
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }

    return data(
      {
        success: false,
        error: "Invalid action specified",
      },
      { status: 400 },
    );
  } catch (error) {
    logger.error("Error processing datapoint action:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);

    // Check if it's a JSON parsing error
    if (errorMessage.includes("JSON")) {
      return data(
        {
          success: false,
          error: `Invalid JSON format: ${errorMessage}. Please ensure all JSON fields contain valid JSON.`,
        },
        { status: 400 },
      );
    }

    return data(
      {
        success: false,
        error: errorMessage,
      },
      { status: 400 },
    );
  }
}

export const handle: RouteHandle = {
  crumb: (match) => [match.params.id!],
};

export async function loader({
  params,
}: {
  params: { dataset_name: string; id: string };
}) {
  const { dataset_name, id } = params;
  if (!dataset_name || !id) {
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }
  const datapoint = await getDatapoint(dataset_name, id);
  if (!datapoint) {
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }
  return {
    datapoint,
  };
}

export default function DatapointPage({ loaderData }: Route.ComponentProps) {
  const { datapoint } = loaderData;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const [input, setInput] = useState<typeof datapoint.input>(datapoint.input);
  const [originalInput] = useState(datapoint.input);
  const [originalOutput] = useState(datapoint.output);
  const [originalTags] = useState(datapoint.tags || {});
  const [output, setOutput] = useState<
    ContentBlockChatOutput[] | JsonInferenceOutput | null
  >(datapoint.output ?? null);
  const [tags, setTags] = useState<Record<string, string>>(
    datapoint.tags || {},
  );
  const [isEditing, setIsEditing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const canSave = useMemo(() => {
    // Use JSON.stringify to compare object values rather than references
    const hasInputChanged =
      JSON.stringify(input) !== JSON.stringify(originalInput);
    const hasOutputChanged =
      JSON.stringify(output) !== JSON.stringify(originalOutput);
    const hasTagsChanged =
      JSON.stringify(tags) !== JSON.stringify(originalTags);

    return isEditing && (hasInputChanged || hasOutputChanged || hasTagsChanged);
  }, [
    isEditing,
    input,
    output,
    tags,
    originalInput,
    originalOutput,
    originalTags,
  ]);

  const toggleEditing = () => setIsEditing(!isEditing);

  const handleReset = () => {
    setInput(datapoint.input);
    setOutput(datapoint.output ?? null);
    setTags(datapoint.tags || {});
  };

  const handleSystemChange = (system: string | object) =>
    setInput({ ...input, system });

  const handleMessagesChange = (messages: DisplayInputMessage[]) => {
    setInput({ ...input, messages });
  };

  const fetcher = useFetcher();
  const saveError = fetcher.data?.success === false ? fetcher.data.error : null;

  const submitDatapointAction = (action: string) => {
    const formData = new FormData();

    // Create a copy of datapoint with updated input, output, and tags if we're saving
    const dataToSubmit = { ...datapoint, input, output, tags };

    Object.entries(dataToSubmit).forEach(([key, value]) => {
      if (value === undefined) return;
      if (value === null) {
        // do nothing
      } else if (typeof value === "object") {
        formData.append(key, JSON.stringify(value));
      } else {
        formData.append(key, String(value));
      }
    });

    formData.append("action", action);

    // Submit to the local action by targeting the current route (".")
    fetcher.submit(formData, { method: "post", action: "." });
  };

  const handleDelete = () => submitDatapointAction("delete");
  const handleSave = () => {
    // Clear any previous validation errors
    setValidationError(null);

    // Validate JSON output before submitting
    if (output && "raw" in output && output.raw) {
      try {
        JSON.parse(output.raw);
      } catch {
        setValidationError(
          "Invalid JSON in output. Please fix the JSON format before saving.",
        );
        return;
      }
    }

    submitDatapointAction("save");
    if (!saveError) {
      setIsEditing(false);
    }
  };

  const functionConfig = useFunctionConfig(datapoint.function_name);
  const variants = Object.keys(functionConfig?.variants || {});

  const variantInferenceFetcher = useInferenceActionFetcher();
  const variantSource = "datapoint";
  const variantInferenceIsLoading =
    // only concerned with rendering loading state when the modal is open
    isModalOpen &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
    const request = prepareInferenceActionRequest({
      resource: datapoint,
      source: variantSource,
      variant,
    });
    // TODO: handle JSON.stringify error
    submit({ data: JSON.stringify(request) });
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
  };

  return (
    <PageLayout>
      <PageHeader
        label="Datapoint"
        name={datapoint.id}
        tag={datapoint.is_custom && <Badge className="ml-2">Custom</Badge>}
      />

      {(saveError || validationError) && (
        <div className="mt-2 rounded-md bg-red-100 px-4 py-3 text-red-800">
          <p className="font-medium">Error saving datapoint</p>
          <p>{validationError || saveError}</p>
        </div>
      )}

      <SectionsGroup>
        <SectionLayout>
          <DatapointBasicInfo datapoint={datapoint} />
        </SectionLayout>

        <SectionLayout>
          <DatapointActions
            variants={variants}
            onVariantSelect={onVariantSelect}
            variantInferenceIsLoading={variantInferenceIsLoading}
            onDelete={handleDelete}
            isDeleting={fetcher.state === "submitting" && !saveError}
            toggleEditing={toggleEditing}
            isEditing={isEditing}
            canSave={canSave}
            onSave={handleSave}
            onReset={handleReset}
            showTryWithVariant={
              datapoint.function_name !== "tensorzero::default"
            }
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputSnippet
            system={input.system}
            messages={input.messages}
            isEditing={isEditing}
            onSystemChange={handleSystemChange}
            onMessagesChange={handleMessagesChange}
          />
        </SectionLayout>

        {output && (
          <SectionLayout>
            <SectionHeader heading="Output" />
            <Output
              output={output}
              isEditing={isEditing}
              onOutputChange={(output) => {
                setOutput(output);
                // Clear validation error when output changes
                setValidationError(null);
              }}
            />
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Tags" />
          <TagsTable tags={tags} onTagsChange={setTags} isEditing={isEditing} />
        </SectionLayout>
      </SectionsGroup>

      {selectedVariant && (
        <VariantResponseModal
          isOpen={isModalOpen}
          isLoading={variantInferenceIsLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={handleModalClose}
          item={datapoint}
          selectedVariant={selectedVariant}
          source="datapoint"
        />
      )}
    </PageLayout>
  );
}

function getUserFacingError(error: unknown): {
  heading: string;
  message: ReactNode;
} {
  if (isRouteErrorResponse(error)) {
    switch (error.status) {
      case 400:
        return {
          heading: `${error.status}: Bad Request`,
          message: "Please try again later.",
        };
      case 401:
        return {
          heading: `${error.status}: Unauthorized`,
          message: "You do not have permission to access this resource.",
        };
      case 403:
        return {
          heading: `${error.status}: Forbidden`,
          message: "You do not have permission to access this resource.",
        };
      case 404:
        return {
          heading: `${error.status}: Not Found`,
          message:
            "The requested resource was not found. Please check the URL and try again.",
        };
      case 500:
      default:
        return {
          heading: "An unknown error occurred",
          message: "Please try again later.",
        };
    }
  }
  return {
    heading: "An unknown error occurred",
    message: "Please try again later.",
  };
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  useEffect(() => {
    logger.error(error);
  }, [error]);
  const { heading, message } = getUserFacingError(error);
  const { dataset_name: datasetName } = useParams<{
    dataset_name: string;
    id: string;
  }>();
  return (
    <div className="flex flex-col items-center justify-center md:h-full">
      <div className="mt-8 flex flex-col items-center justify-center gap-2 rounded-xl bg-red-50 p-6 md:mt-0">
        <h1 className="text-2xl font-bold">{heading}</h1>
        {typeof message === "string" ? <p>{message}</p> : message}
        <Link
          to={`/datasets/${datasetName}`}
          className="font-bold text-red-800 hover:text-red-600"
        >
          Go back &rarr;
        </Link>
      </div>
    </div>
  );
}

function transformOutputForTensorZero(
  output: ParsedDatasetRow["output"],
): string | null {
  if (output === null || output === undefined) {
    return null;
  } else if ("raw" in output) {
    if (output.raw === null) {
      return null;
    }
    try {
      return JSON.parse(output.raw);
    } catch (error) {
      throw new Error(
        `Invalid JSON in output.raw: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  } else if (typeof output === "object") {
    try {
      return JSON.parse(JSON.stringify(output));
    } catch (error) {
      throw new Error(
        `Failed to serialize output object: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  } else {
    return output;
  }
}
