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
import BasicInfo from "./DatapointBasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useConfig } from "~/context/config";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import type { Route } from "./+types/route";
import {
  ParsedDatasetRowSchema,
  type ParsedDatasetRow,
} from "~/utils/clickhouse/datasets";
import {
  staleDatapoint,
  getDatasetCounts,
} from "~/utils/clickhouse/datasets.server";
import { tensorZeroClient } from "~/utils/tensorzero.server";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { DatapointActions } from "./DatapointActions";
import type { ResolvedInputMessage } from "~/utils/clickhouse/common";
import { getConfig } from "~/utils/config/index.server";
import { resolvedInputToTensorZeroInput } from "~/routes/api/tensorzero/inference";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

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
  };

  const cleanedData = Object.fromEntries(
    Object.entries(rawData).filter(([, value]) => value !== undefined),
  );
  const parsedFormData: ParsedDatasetRow =
    ParsedDatasetRowSchema.parse(cleanedData);
  const config = await getConfig();
  const functionConfig = config.functions[parsedFormData.function_name];
  const functionType = functionConfig?.type;
  const action = formData.get("action");
  if (action === "delete") {
    await staleDatapoint(
      parsedFormData.dataset_name,
      parsedFormData.id,
      functionType,
    );
    const datasetCounts = await getDatasetCounts();
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

      const datapoint = {
        function_name: parsedFormData.function_name,
        input: transformedInput,
        output: transformedOutput,
        tags: parsedFormData.tags || {},
        auxiliary: parsedFormData.auxiliary,
        ...(functionType === "json"
          ? {
              output_schema:
                parsedFormData["output_schema" as keyof typeof parsedFormData],
            }
          : {}),
        ...(functionType === "chat" && "tool_params" in parsedFormData
          ? {
              tool_params:
                parsedFormData["tool_params" as keyof typeof parsedFormData],
            }
          : {}),
        source_inference_id: parsedFormData.source_inference_id,
      };
      const { id } = await tensorZeroClient.updateDatapoint(
        parsedFormData.dataset_name,
        uuid(),
        datapoint,
        formData.get("inputChanged") === "true",
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
      console.error("Error updating datapoint:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
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
  const [output, setOutput] = useState<typeof datapoint.output>(
    datapoint.output,
  );
  const config = useConfig();
  const [isEditing, setIsEditing] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [canSave, setCanSave] = useState(false);
  const [inputChanged, setInputChanged] = useState(false);
  const [outputChanged, setOutputChanged] = useState(false);

  useEffect(() => {
    // Use JSON.stringify to compare object values rather than references
    const hasInputChanged =
      JSON.stringify(input) !== JSON.stringify(originalInput);
    const hasOutputChanged =
      JSON.stringify(output) !== JSON.stringify(originalOutput);

    setInputChanged(hasInputChanged);
    setOutputChanged(hasOutputChanged);
    setCanSave(isEditing && (hasInputChanged || hasOutputChanged));
  }, [isEditing, input, output, originalInput, originalOutput]);

  const toggleEditing = () => setIsEditing(!isEditing);

  const handleReset = () => {
    setInput(datapoint.input);
    setOutput(datapoint.output);
    setResetKey((prev) => prev + 1);
  };

  const handleSystemChange = (system: string | object) => {
    setInput({ ...input, system });
  };

  const handleMessagesChange = (messages: ResolvedInputMessage[]) => {
    setInput({ ...input, messages });
  };

  const handleOutputChange = (newOutput: typeof datapoint.output | null) => {
    if (newOutput === null) {
      setCanSave(false);
    } else {
      const hasOutputChanged =
        JSON.stringify(newOutput) !== JSON.stringify(originalOutput);
      const hasInputChanged =
        JSON.stringify(input) !== JSON.stringify(originalInput);
      setOutput(newOutput);
      setCanSave(isEditing && (hasOutputChanged || hasInputChanged));
    }
  };

  const fetcher = useFetcher();
  const saveError = fetcher.data?.success === false ? fetcher.data.error : null;

  const submitDatapointAction = (action: string) => {
    const formData = new FormData();

    // Create a copy of datapoint with updated input and output if we're saving
    const dataToSubmit = { ...datapoint, input, output };

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
    formData.append("inputChanged", String(inputChanged));
    formData.append("outputChanged", String(outputChanged));

    // Submit to the local action by targeting the current route (".")
    fetcher.submit(formData, { method: "post", action: "." });
  };

  const handleDelete = () => submitDatapointAction("delete");
  const handleSave = () => {
    submitDatapointAction("save");
    if (!saveError) {
      setIsEditing(false);
    }
  };

  const variants = Object.keys(
    config.functions[datapoint.function_name]?.variants || {},
  );

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
      <PageHeader label="Datapoint" name={datapoint.id} />
      {saveError && (
        <div className="mt-2 rounded-md bg-red-100 px-4 py-3 text-red-800">
          <p className="font-medium">Error saving datapoint</p>
          <p>{saveError}</p>
        </div>
      )}

      <SectionsGroup>
        <SectionLayout>
          <BasicInfo datapoint={datapoint} />
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
          <Input
            key={`input-${resetKey}`}
            input={input}
            isEditing={isEditing}
            onSystemChange={handleSystemChange}
            onMessagesChange={handleMessagesChange}
          />
        </SectionLayout>

        {output && (
          <SectionLayout>
            <SectionHeader heading="Output" />
            <Output
              key={`output-${resetKey}`}
              output={output}
              isEditing={isEditing}
              onOutputChange={handleOutputChange}
            />
          </SectionLayout>
        )}
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
    console.error(error);
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
    return JSON.parse(output.raw);
  } else if (typeof output === "object") {
    return JSON.parse(JSON.stringify(output));
  } else {
    return output;
  }
}
