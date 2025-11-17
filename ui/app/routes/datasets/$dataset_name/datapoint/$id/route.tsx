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
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { logger } from "~/utils/logger";
import { resolveStoredInputToInput } from "~/utils/resolve.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { Route } from "./+types/route";
import { DatapointActions } from "./DatapointActions";
import DatapointBasicInfo from "./DatapointBasicInfo";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
  Input,
} from "~/types/tensorzero";
import {
  deleteDatapoint,
  renameDatapoint,
  updateDatapoint,
} from "./datapointOperations.server";
import {
  parseDatapointAction,
  serializeDeleteDatapointToFormData,
  serializeUpdateDatapointToFormData,
  serializeRenameDatapointToFormData,
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

export function hasDatapointChanged(params: {
  currentInput: Input;
  originalInput: Input;
  currentOutput?: ContentBlockChatOutput[] | JsonInferenceOutput;
  originalOutput: ContentBlockChatOutput[] | JsonInferenceOutput | undefined;
  currentTags: Record<string, string>;
  originalTags: Record<string, string>;
}): boolean {
  const {
    currentInput,
    originalInput,
    currentOutput,
    originalOutput,
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
  const hasTagsChanged =
    JSON.stringify(currentTags) !== JSON.stringify(originalTags);

  return hasInputChanged || hasOutputChanged || hasTagsChanged;
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
  const t0Datapoint = await getTensorZeroClient().getDatapoint(id);
  if (!t0Datapoint) {
    throw data(`No datapoint found for ID \`${id}\`.`, {
      status: 404,
    });
  }
  // Note (GabrielBianconi): `getDatapoint` no longer depends on the dataset name, but maybe it should?
  if (t0Datapoint.dataset_name !== dataset_name) {
    throw data(
      `The datapoint \`${id}\` does not belong to dataset \`${dataset_name}\`.`,
      {
        status: 400,
      },
    );
  }
  // Resolve input for InputElement component
  const resolvedInput = await resolveStoredInputToInput(t0Datapoint.input);

  return {
    t0Datapoint,
    resolvedInput,
  };
}

export default function DatapointPage({ loaderData }: Route.ComponentProps) {
  const { t0Datapoint, resolvedInput } = loaderData;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  const [originalInput, setOriginalInput] = useState(resolvedInput);
  const [input, setInput] = useState<Input>(resolvedInput);

  const [originalOutput, setOriginalOutput] = useState(t0Datapoint.output);
  const [output, setOutput] = useState(t0Datapoint.output);

  const [originalTags, setOriginalTags] = useState(t0Datapoint.tags || {});
  const [tags, setTags] = useState(t0Datapoint.tags || {});

  const [isEditing, setIsEditing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Reset state when datapoint changes (e.g., after save redirect)
  useEffect(() => {
    setInput(resolvedInput);
    setOriginalInput(resolvedInput);
    setOutput(t0Datapoint.output);
    setOriginalOutput(t0Datapoint.output);
    setTags(t0Datapoint.tags || {});
    setOriginalTags(t0Datapoint.tags || {});
    setIsEditing(false);
    setValidationError(null);
  }, [resolvedInput, t0Datapoint]);

  const canSave = useMemo(() => {
    return (
      isEditing &&
      hasDatapointChanged({
        currentInput: input,
        originalInput,
        currentOutput: output,
        originalOutput,
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
    setOutput(t0Datapoint.output);
    setTags(t0Datapoint.tags || {});
  };

  const fetcher = useFetcher();
  const updateError =
    fetcher.data?.success === false ? fetcher.data.error : null;

  const handleDelete = () => {
    try {
      const formData = serializeDeleteDatapointToFormData({
        dataset_name: t0Datapoint.dataset_name,
        id: t0Datapoint.id,
      });
      fetcher.submit(formData, { method: "post", action: "." });
    } catch (error) {
      logger.error("Error preparing delete request:", error);
    }
  };

  const handleUpdate = () => {
    setValidationError(null);

    const validation = validateJsonOutput(output);
    if (!validation.valid) {
      setValidationError(validation.error);
      return;
    }

    try {
      const formData = serializeUpdateDatapointToFormData({
        dataset_name: t0Datapoint.dataset_name,
        function_name: t0Datapoint.function_name,
        id: t0Datapoint.id,
        episode_id: t0Datapoint.episode_id,
        input,
        output,
        tags,
      });
      fetcher.submit(formData, { method: "post", action: "." });
      // Note: Edit mode will be exited by the useEffect when the datapoint updates on success
    } catch (error) {
      logger.error("Error preparing update request:", error);
    }
  };

  const functionConfig = useFunctionConfig(t0Datapoint.function_name);
  const variants = Object.keys(functionConfig?.variants || {});

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
    // TODO: error handling
    const request = prepareInferenceActionRequest(args);
    if (bypassCache) {
      request.cache_options = {
        ...request.cache_options,
        enabled: "write_only",
      };
    }
    setLastRequestArgs(args);
    submit({ data: JSON.stringify(request) });
  };

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
    submitVariantInference({
      resource: t0Datapoint,
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
        dataset_name: t0Datapoint.dataset_name,
        id: t0Datapoint.id,
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
        name={t0Datapoint.id}
        tag={
          <>
            {t0Datapoint.is_custom && (
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
            {t0Datapoint.staled_at && (
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
            datapoint={t0Datapoint}
            onRenameDatapoint={handleRenameDatapoint}
          />
        </SectionLayout>

        <SectionLayout>
          <DatapointActions
            variants={variants}
            onVariantSelect={onVariantSelect}
            variantInferenceIsLoading={variantInferenceIsLoading}
            onDelete={handleDelete}
            isDeleting={fetcher.state === "submitting" && !updateError}
            toggleEditing={toggleEditing}
            isEditing={isEditing}
            canSave={canSave}
            onSave={handleUpdate}
            onReset={handleReset}
            showTryWithButton={t0Datapoint.function_name !== DEFAULT_FUNCTION}
            isStale={!!t0Datapoint.staled_at}
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
          item={t0Datapoint}
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
