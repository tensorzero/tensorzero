import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import {
  data,
  isRouteErrorResponse,
  Link,
  redirect,
  useFetcher,
  useParams,
} from "react-router";
import { toDatapointUrl, toDatasetUrl } from "~/utils/urls";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { TagsTable } from "~/components/tags/TagsTable";
import { useFunctionConfig } from "~/context/config";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";
import { useToast } from "~/hooks/use-toast";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { logger } from "~/utils/logger";
import { loadFileDataForInput } from "~/utils/resolve.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { validateJsonSchema } from "~/utils/jsonschema";
import type { Route } from "./+types/route";
import { DatapointActions } from "./DatapointActions";
import DatapointBasicInfo from "./DatapointBasicInfo";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
  Input,
  Datapoint,
  JsonValue,
} from "~/types/tensorzero";

// Discriminated union for type-safe output state
type OutputState =
  | { type: "chat"; value?: ContentBlockChatOutput[] }
  | { type: "json"; value?: JsonInferenceOutput; outputSchema: JsonValue };

function createOutputState(datapoint: Datapoint): OutputState {
  switch (datapoint.type) {
    case "chat":
      return {
        type: "chat",
        value: datapoint.output as ContentBlockChatOutput[] | undefined,
      };
    case "json":
      return {
        type: "json",
        value: datapoint.output as JsonInferenceOutput | undefined,
        outputSchema: datapoint.output_schema,
      };
  }
}
import {
  cloneDatapoint,
  deleteDatapoint,
  renameDatapoint,
  updateDatapoint,
} from "./datapointOperations.server";
import {
  parseDatapointAction,
  serializeDeleteDatapointToFormData,
  serializeUpdateDatapointToFormData,
  serializeRenameDatapointToFormData,
  type CloneDatapointFormData,
  type DeleteDatapointFormData,
  type UpdateDatapointFormData,
  type RenameDatapointFormData,
  type DatapointAction,
} from "./formDataUtils";
import { z } from "zod";

export function validateJsonOutput(
  output?: ContentBlockChatOutput[] | JsonInferenceOutput,
): { valid: true } | { valid: false; error: string } {
  if (output && "raw" in output && output.raw) {
    try {
      JSON.parse(output.raw);
      return { valid: true };
    } catch {
      return {
        valid: false,
        error:
          "Invalid JSON in output. Please fix the JSON format before saving.",
      };
    }
  }
  return { valid: true };
}

export function validateInput(
  _input: Input,
): { valid: true } | { valid: false; error: string } {
  // TODO (#4903): Handle invalid intermediate states; right it'll keep stale version (but there is a visual cue)
  return { valid: true };
}

export function hasDatapointChanged(params: {
  currentInput: Input;
  originalInput: Input;
  currentOutput?: ContentBlockChatOutput[] | JsonInferenceOutput;
  originalOutput?: ContentBlockChatOutput[] | JsonInferenceOutput;
  currentOutputSchema?: JsonValue;
  originalOutputSchema?: JsonValue;
  currentTags: Record<string, string>;
  originalTags: Record<string, string>;
}): boolean {
  const {
    currentInput,
    originalInput,
    currentOutput,
    originalOutput,
    currentOutputSchema,
    originalOutputSchema,
    currentTags,
    originalTags,
  } = params;

  // Check if system has changed (added, removed, or modified)
  const hasSystemChanged =
    "system" in currentInput !== "system" in originalInput ||
    JSON.stringify(currentInput.system) !==
      JSON.stringify(originalInput.system);

  // Check if messages changed
  const hasMessagesChanged =
    JSON.stringify(currentInput.messages) !==
    JSON.stringify(originalInput.messages);

  const hasInputChanged = hasSystemChanged || hasMessagesChanged;

  const hasOutputChanged =
    JSON.stringify(currentOutput) !== JSON.stringify(originalOutput);
  const hasOutputSchemaChanged =
    JSON.stringify(currentOutputSchema) !==
    JSON.stringify(originalOutputSchema);
  const hasTagsChanged =
    JSON.stringify(currentTags) !== JSON.stringify(originalTags);

  return (
    hasInputChanged ||
    hasOutputChanged ||
    hasOutputSchemaChanged ||
    hasTagsChanged
  );
}

async function handleDeleteAction(
  actionData: DeleteDatapointFormData,
): Promise<
  Response | ReturnType<typeof data<{ success: boolean; error: string }>>
> {
  try {
    const { redirectTo } = await deleteDatapoint({
      dataset_name: actionData.dataset_name,
      id: actionData.id,
    });
    return redirect(redirectTo);
  } catch (error) {
    logger.error("Error deleting datapoint:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}

async function handleUpdateAction(
  actionData: UpdateDatapointFormData,
): Promise<
  Response | ReturnType<typeof data<{ success: boolean; error: string }>>
> {
  let functionConfig;
  try {
    const config = await getConfig();
    functionConfig = await getFunctionConfig(actionData.function_name, config);

    if (!functionConfig) {
      return data(
        {
          success: false,
          error: `Failed to find function config for function ${actionData.function_name}`,
        },
        { status: 400 },
      );
    }
  } catch (error) {
    logger.error("Error fetching function config:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }

  const functionType = functionConfig.type;

  if (actionData.output) {
    const validation = validateJsonOutput(actionData.output);
    if (!validation.valid) {
      return data(
        {
          success: false,
          error: validation.error,
        },
        { status: 400 },
      );
    }
  }

  try {
    const { newId } = await updateDatapoint({
      parsedFormData: {
        dataset_name: actionData.dataset_name,
        function_name: actionData.function_name,
        id: actionData.id,
        episode_id: actionData.episode_id,
        input: actionData.input,
        output: actionData.output,
        output_schema: actionData.output_schema,
        tags: actionData.tags,
      },
      functionType,
    });
    return redirect(toDatapointUrl(actionData.dataset_name, newId));
  } catch (error) {
    logger.error("Error updating datapoint:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}

async function handleRenameAction(
  actionData: RenameDatapointFormData,
): Promise<ReturnType<typeof data<{ success: boolean; error?: string }>>> {
  const nameToSet = actionData.name === "" ? null : actionData.name;

  try {
    await renameDatapoint({
      datasetName: actionData.dataset_name,
      datapointId: actionData.id,
      name: nameToSet,
    });
    return data({ success: true });
  } catch (error) {
    logger.error("Error renaming datapoint:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}

async function handleCloneAction(
  actionData: CloneDatapointFormData,
): Promise<
  ReturnType<
    typeof data<{ success: boolean; error?: string; redirectTo?: string }>
  >
> {
  try {
    const datapointData = JSON.parse(actionData.datapoint) as Datapoint;
    const { newId } = await cloneDatapoint({
      targetDataset: actionData.target_dataset,
      datapoint: datapointData,
    });
    return data({
      success: true,
      redirectTo: toDatapointUrl(actionData.target_dataset, newId),
    });
  } catch (error) {
    logger.error("Error cloning datapoint:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  let parsedAction: DatapointAction;
  try {
    parsedAction = parseDatapointAction(formData);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return data(
        {
          success: false,
          error: `Validation failed: ${error.errors.map((e) => e.message).join(", ")}`,
        },
        { status: 400 },
      );
    }
    logger.error("Error parsing datapoint action:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 400 },
    );
  }

  switch (parsedAction.action) {
    case "delete":
      return handleDeleteAction(parsedAction);
    case "update":
      return handleUpdateAction(parsedAction);
    case "rename":
      return handleRenameAction(parsedAction);
    case "clone":
      return handleCloneAction(parsedAction);
  }
}

export const handle: RouteHandle = {
  crumb: (match) => [
    "Datapoints",
    { label: match.params.id!, isIdentifier: true },
  ],
};

export async function loader({
  params,
}: {
  params: { dataset_name: string; id: string };
}) {
  const { dataset_name, id } = params;
  if (!dataset_name || !id) {
    throw data("You must provide a dataset name and datapoint ID.", {
      status: 404,
    });
  }
  const datapoint = await getTensorZeroClient().getDatapoint(id, dataset_name);
  if (!datapoint) {
    throw data(`No datapoint found for ID \`${id}\`.`, {
      status: 404,
    });
  }

  // Load file data for InputElement component
  const resolvedInput = await loadFileDataForInput(datapoint.input);

  return {
    datapoint,
    resolvedInput,
  };
}

export default function DatapointPage({ loaderData }: Route.ComponentProps) {
  const { datapoint, resolvedInput } = loaderData;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  // State Management for Editing
  //
  // We keep track of the original values and current values in the form (when editing).
  // This allows us to detect if there were any changes (to enable/disable the save button) or discard changes if the user cancels the edits
  // When you save, we check `XXX` (`input`, `output`, `tags`) against `originalXXX` and submit `XXX` to the update datapoint endpoint.
  const [originalInput, setOriginalInput] = useState(resolvedInput);
  const [input, setInput] = useState<Input>(resolvedInput);

  const [originalOutput, setOriginalOutput] = useState<OutputState>(() =>
    createOutputState(datapoint),
  );
  const [output, setOutput] = useState<OutputState>(() =>
    createOutputState(datapoint),
  );

  const [originalTags, setOriginalTags] = useState(datapoint.tags || {});
  const [tags, setTags] = useState(datapoint.tags || {});

  const [isEditing, setIsEditing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Reset state when datapoint changes (e.g., after save redirect)
  useEffect(() => {
    setInput(resolvedInput);
    setOriginalInput(resolvedInput);
    const newOutputState = createOutputState(datapoint);
    setOutput(newOutputState);
    setOriginalOutput(newOutputState);
    setTags(datapoint.tags || {});
    setOriginalTags(datapoint.tags || {});
    setIsEditing(false);
    setValidationError(null);
  }, [resolvedInput, datapoint]);

  const canSave = useMemo(() => {
    const currentOutputSchema =
      output.type === "json" ? output.outputSchema : undefined;
    const origOutputSchema =
      originalOutput.type === "json" ? originalOutput.outputSchema : undefined;

    return (
      isEditing &&
      hasDatapointChanged({
        currentInput: input,
        originalInput,
        currentOutput: output.value,
        originalOutput: originalOutput.value,
        currentOutputSchema,
        originalOutputSchema: origOutputSchema,
        currentTags: tags,
        originalTags,
      })
    );
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
    setInput(resolvedInput);
    setOutput(createOutputState(datapoint));
    setTags(datapoint.tags || {});
  };

  const fetcher = useFetcher();
  const updateError =
    fetcher.data?.success === false ? fetcher.data.error : null;

  // Determine which action is being performed by checking the form data
  // Include both "submitting" and "loading" states to keep spinner visible until page updates
  const pendingAction = fetcher.formData?.get("action") as string | null;
  const isSubmittingOrLoading =
    fetcher.state === "submitting" || fetcher.state === "loading";
  const isSaving = isSubmittingOrLoading && pendingAction === "update";
  const isDeleting = isSubmittingOrLoading && pendingAction === "delete";

  const handleDelete = () => {
    try {
      const formData = serializeDeleteDatapointToFormData({
        dataset_name: datapoint.dataset_name,
        id: datapoint.id,
      });
      fetcher.submit(formData, { method: "post", action: "." });
    } catch (error) {
      logger.error("Error preparing delete request:", error);
    }
  };

  const handleUpdate = () => {
    setValidationError(null);

    const outputValidation = validateJsonOutput(output.value);
    if (!outputValidation.valid) {
      setValidationError(outputValidation.error);
      return;
    }

    const inputValidation = validateInput(input);
    if (!inputValidation.valid) {
      setValidationError(inputValidation.error);
      return;
    }

    // Validate schema for JSON output
    if (output.type === "json") {
      const schemaValidation = validateJsonSchema(output.outputSchema);
      if (!schemaValidation.valid) {
        setValidationError(schemaValidation.error);
        return;
      }
    }

    try {
      const formData = serializeUpdateDatapointToFormData({
        dataset_name: datapoint.dataset_name,
        function_name: datapoint.function_name,
        id: datapoint.id,
        episode_id: datapoint.episode_id,
        input,
        output: output.value,
        output_schema: output.type === "json" ? output.outputSchema : undefined,
        tags,
      });
      fetcher.submit(formData, { method: "post", action: "." });
      // Note: Edit mode will be exited by the useEffect when the datapoint updates on success
    } catch (error) {
      logger.error("Error preparing update request:", error);
    }
  };

  const functionConfig = useFunctionConfig(datapoint.function_name);
  const variants = Object.keys(functionConfig?.variants || {});

  const { toast } = useToast();
  const variantInferenceFetcher = useInferenceActionFetcher();
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);

  const variantInferenceIsLoading =
    // only concerned with rendering loading state when the modal is open
    isModalOpen &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const submitVariantInference = (
    args: Parameters<typeof prepareInferenceActionRequest>[0],
    { bypassCache }: { bypassCache?: boolean } = {},
  ) => {
    try {
      const request = prepareInferenceActionRequest(args);
      if (bypassCache) {
        request.cache_options = {
          ...request.cache_options,
          enabled: "write_only",
        };
      }
      setLastRequestArgs(args);

      try {
        submit({ data: JSON.stringify(request) });
      } catch (stringifyError) {
        logger.error("Failed to stringify request:", stringifyError);
        toast.error({
          title: "Request Error",
          description: "Failed to prepare the request. Please try again.",
        });
        // Reset state on error
        setLastRequestArgs(null);
        setIsModalOpen(false);
        setSelectedVariant(null);
      }
    } catch (error) {
      logger.error("Failed to prepare inference request:", error);

      // Show user-friendly error message based on the error type
      let errorMessage = "Failed to prepare the request. Please try again.";
      if (error instanceof Error) {
        if (error.message.includes("Extra body is not supported")) {
          errorMessage =
            "This datapoint contains extra body parameters which are not supported in the UI.";
        } else if (error.message) {
          errorMessage = error.message;
        }
      }

      toast.error({
        title: "Request Preparation Error",
        description: errorMessage,
      });
    }
  };

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
    submitVariantInference({
      resource: datapoint,
      source: "t0_datapoint",
      variant,
    });
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setLastRequestArgs(null);
  };

  const handleRefresh = () => {
    if (!lastRequestArgs) {
      return;
    }
    submitVariantInference(lastRequestArgs, { bypassCache: true });
  };

  const handleRenameDatapoint = async (newName: string) => {
    try {
      const formData = serializeRenameDatapointToFormData({
        dataset_name: datapoint.dataset_name,
        id: datapoint.id,
        name: newName,
      });
      await fetcher.submit(formData, { method: "post", action: "." });
    } catch (error) {
      logger.error("Error preparing rename request:", error);
    }
  };

  return (
    <PageLayout>
      <PageHeader
        label="Datapoint"
        name={datapoint.id}
        tag={
          <>
            {datapoint.is_custom && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="ml-2 cursor-help">
                    Custom
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  This datapoint is not based on a historical inference. It was
                  either edited or created manually.
                </TooltipContent>
              </Tooltip>
            )}
            {datapoint.staled_at && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="ml-2 cursor-help">
                    Stale
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  This datapoint has since been edited or deleted.
                </TooltipContent>
              </Tooltip>
            )}
          </>
        }
      />

      {(updateError || validationError) && (
        <div className="mt-2 rounded-md bg-red-100 px-4 py-3 text-red-800">
          <p className="font-medium">Error updating datapoint</p>
          <p>{validationError || updateError}</p>
        </div>
      )}

      <SectionsGroup>
        <SectionLayout>
          <DatapointBasicInfo
            datapoint={datapoint}
            onRenameDatapoint={handleRenameDatapoint}
          />
        </SectionLayout>

        <SectionLayout>
          <DatapointActions
            variants={variants}
            onVariantSelect={onVariantSelect}
            variantInferenceIsLoading={variantInferenceIsLoading}
            onDelete={handleDelete}
            isDeleting={isDeleting}
            toggleEditing={toggleEditing}
            isEditing={isEditing}
            canSave={canSave}
            isSaving={isSaving}
            onSave={handleUpdate}
            onReset={handleReset}
            showTryWithButton={datapoint.function_name !== DEFAULT_FUNCTION}
            isStale={!!datapoint.staled_at}
            datapoint={datapoint}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputElement
            input={input}
            isEditing={isEditing}
            onSystemChange={(system) => setInput({ ...input, system })}
            onMessagesChange={(messages) => setInput({ ...input, messages })}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          {(() => {
            switch (output.type) {
              case "chat":
                return (
                  <ChatOutputElement
                    output={output.value}
                    isEditing={isEditing}
                    onOutputChange={(value) => {
                      setOutput({ type: "chat", value });
                      setValidationError(null);
                    }}
                  />
                );
              case "json":
                return (
                  <JsonOutputElement
                    output={output.value}
                    outputSchema={output.outputSchema}
                    isEditing={isEditing}
                    onOutputChange={(value) => {
                      setOutput({
                        type: "json",
                        value,
                        outputSchema: output.outputSchema,
                      });
                      setValidationError(null);
                    }}
                    onOutputSchemaChange={(outputSchema) => {
                      setOutput({
                        type: "json",
                        value: output.value,
                        outputSchema,
                      });
                      setValidationError(null);
                    }}
                  />
                );
            }
          })()}
        </SectionLayout>

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
          onRefresh={lastRequestArgs ? handleRefresh : null}
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
        {datasetName && (
          <Link
            to={toDatasetUrl(datasetName)}
            className="font-bold text-red-800 hover:text-red-600"
          >
            Go back &rarr;
          </Link>
        )}
      </div>
    </div>
  );
}
