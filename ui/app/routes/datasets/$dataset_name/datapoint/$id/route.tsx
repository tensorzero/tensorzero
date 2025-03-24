import { useFetcher } from "react-router";
import { data, isRouteErrorResponse, redirect } from "react-router";
import { v7 as uuid } from "uuid";
import BasicInfo from "./DatapointBasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import { useEffect, useState, useMemo } from "react";
import { useConfig } from "~/context/config";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { VariantResponseModal } from "./VariantResponseModal";
import type { Route } from "./+types/route";
import type { ActionFunctionArgs } from "react-router";
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
import type {
  Input as ClickHouseInput,
  InputMessage,
  InputMessageContent,
} from "~/utils/clickhouse/common";
import { getConfig } from "~/utils/config/index.server";

/**
 * Transforms input from clickhouse format to TensorZero client format
 */
function transformInputForTensorZero(input: ClickHouseInput) {
  return {
    system: input.system,
    messages: input.messages.map((msg: InputMessage) => ({
      role: msg.role as "system" | "user" | "assistant" | "tool",
      content: msg.content.map((c: InputMessageContent) => {
        if (c.type === "text") {
          return { type: "text" as const, value: c.value };
        } else if (c.type === "tool_call") {
          return {
            type: "tool_call" as const,
            id: c.id,
            name: c.name,
            arguments: c.arguments,
          };
        } else if (c.type === "tool_result") {
          return {
            type: "tool_result" as const,
            id: c.id,
            name: c.name,
            result: c.result,
          };
        } else if (c.type === "image") {
          // Skip image content as it's not supported in the target schema
          return {
            type: "text" as const,
            value: "[Image content not supported]",
          };
        }
        // Fallback case
        return { type: "text" as const, value: "Unsupported content type" };
      }),
    })),
  };
}

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
  };

  const cleanedData = Object.fromEntries(
    Object.entries(rawData).filter(([, value]) => value !== undefined),
  );
  const parsedFormData: ParsedDatasetRow =
    ParsedDatasetRowSchema.parse(cleanedData);
  const config = await getConfig();
  const functionConfig = config.functions[parsedFormData.function_name];
  const functionType = functionConfig?.type;
  // await deleteDatapointServer(parsedFormData);
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
    // Transform input to match TensorZero client's expected format
    const transformedInput = transformInputForTensorZero(parsedFormData.input);
    const transformedOutput = transformOutputForTensorZero(
      parsedFormData.output,
    );

    try {
      const { id } = await tensorZeroClient.updateDatapoint(
        parsedFormData.dataset_name,
        uuid(),
        {
          function_name: parsedFormData.function_name,
          input: transformedInput,
          output: transformedOutput,
          tags: parsedFormData.tags || {},
          auxiliary: parsedFormData.auxiliary,
          ...(functionType === "json"
            ? {
                output_schema:
                  parsedFormData[
                    "output_schema" as keyof typeof parsedFormData
                  ],
              }
            : {}),
          ...(functionType === "chat" && "tool_params" in parsedFormData
            ? {
                tool_params:
                  parsedFormData["tool_params" as keyof typeof parsedFormData],
              }
            : {}),
        },
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
  const [variantInferenceIsLoading, setVariantInferenceIsLoading] =
    useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const [input, setInput] = useState<typeof datapoint.input>(datapoint.input);
  const originalInput = useMemo(() => datapoint.input, []);
  const [output, setOutput] = useState<typeof datapoint.output>(
    datapoint.output,
  );
  const originalOutput = useMemo(() => datapoint.output, []);
  const config = useConfig();
  const [isEditing, setIsEditing] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [canSave, setCanSave] = useState(false);
  useEffect(() => {
    // Use JSON.stringify to compare object values rather than references
    const inputChanged =
      JSON.stringify(input) !== JSON.stringify(originalInput);
    const outputChanged =
      JSON.stringify(output) !== JSON.stringify(originalOutput);

    setCanSave(isEditing && (inputChanged || outputChanged));
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

  const handleMessagesChange = (messages: InputMessage[]) => {
    setInput({ ...input, messages });
  };

  const handleOutputChange = (newOutput: typeof datapoint.output) => {
    setOutput(newOutput);
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

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setVariantInferenceIsLoading(false);
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
          setIsLoading={setVariantInferenceIsLoading}
          onClose={handleModalClose}
          datapoint={datapoint}
          selectedVariant={selectedVariant}
        />
      )}
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}

function transformOutputForTensorZero(output: ParsedDatasetRow["output"]) {
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
