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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { TagsTable } from "~/components/tags/TagsTable";
import { useFunctionConfig } from "~/context/config";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";
import type { DisplayInputMessage } from "~/utils/clickhouse/common";

import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { logger } from "~/utils/logger";
import type { Route } from "./+types/route";
import { DatapointActions } from "./DatapointActions";
import DatapointBasicInfo from "./DatapointBasicInfo";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
} from "tensorzero-node";
import {
  deleteDatapoint,
  renameDatapoint,
  saveDatapoint,
} from "./datapointOperations.server";
import {
  parseDatapointFormData,
  serializeDatapointToFormData,
} from "./formDataUtils";

export function validateJsonOutput(
  output: ContentBlockChatOutput[] | JsonInferenceOutput | null,
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
  currentInput: ParsedDatasetRow["input"];
  originalInput: ParsedDatasetRow["input"];
  currentOutput: ContentBlockChatOutput[] | JsonInferenceOutput | null;
  originalOutput: ParsedDatasetRow["output"];
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

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  // TODO(shuyangli): Limit the try-catch to a smaller scope so it's clear what we're catching.
  try {
    const parsedFormData = parseDatapointFormData(formData);
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
      const { redirectTo } = await deleteDatapoint({
        dataset_name: parsedFormData.dataset_name,
        id: parsedFormData.id,
        functionType,
      });
      return redirect(redirectTo);
    } else if (action === "save") {
      try {
        const { newId } = await saveDatapoint({
          parsedFormData,
          functionType,
        });
        return redirect(toDatapointUrl(parsedFormData.dataset_name, newId));
      } catch (error) {
        logger.error("Error updating datapoint:", error);
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    } else if (action === "rename") {
      await renameDatapoint({
        functionType: functionType,
        datasetName: parsedFormData.dataset_name,
        datapoint: parsedFormData,
        newName: parsedFormData.name || "",
      });
      return data({ success: true });
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
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }
  const datapoint = await getDatapoint(dataset_name, id, true);
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
  const [originalInput, setOriginalInput] = useState(datapoint.input);
  const [originalOutput, setOriginalOutput] = useState(datapoint.output);
  const [originalTags, setOriginalTags] = useState(datapoint.tags || {});
  const [output, setOutput] = useState<
    ContentBlockChatOutput[] | JsonInferenceOutput | null
  >(datapoint.output ?? null);
  const [tags, setTags] = useState<Record<string, string>>(
    datapoint.tags || {},
  );
  const [isEditing, setIsEditing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Reset state when datapoint changes (e.g., after save redirect)
  useEffect(() => {
    setInput(datapoint.input);
    setOriginalInput(datapoint.input);
    setOutput(datapoint.output ?? null);
    setOriginalOutput(datapoint.output);
    setTags(datapoint.tags || {});
    setOriginalTags(datapoint.tags || {});
    setIsEditing(false);
    setValidationError(null);
  }, [datapoint]);

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
    setInput(datapoint.input);
    setOutput(datapoint.output ?? null);
    setTags(datapoint.tags || {});
  };

  const handleSystemChange = (system: string | object | null) => {
    setInput((prevInput) => {
      if (system === null) {
        // Explicitly create new object without system key
        return {
          messages: prevInput.messages,
        };
      } else {
        return { ...prevInput, system };
      }
    });
  };

  const handleMessagesChange = (messages: DisplayInputMessage[]) => {
    setInput((prevInput) => ({ ...prevInput, messages }));
  };

  const fetcher = useFetcher();
  const saveError = fetcher.data?.success === false ? fetcher.data.error : null;

  const submitDatapointAction = (action: string) => {
    // Create a copy of datapoint with updated input, output, and tags if we're saving
    const dataToSubmit = { ...datapoint, input, output, tags };

    const formData = serializeDatapointToFormData(dataToSubmit);
    formData.append("action", action);

    // Submit to the local action by targeting the current route (".")
    fetcher.submit(formData, { method: "post", action: "." });
  };

  const handleDelete = () => submitDatapointAction("delete");
  const handleSave = () => {
    // Clear any previous validation errors
    setValidationError(null);

    // Validate JSON output before submitting
    const validation = validateJsonOutput(output);
    if (!validation.valid) {
      setValidationError(validation.error);
      return;
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

  const handleRenameDatapoint = async (newName: string) => {
    const dataToSubmit = { ...datapoint, name: newName };
    const formData = serializeDatapointToFormData(dataToSubmit);
    formData.append("action", "rename");
    await fetcher.submit(formData, { method: "post", action: "." });
  };

  return (
    <PageLayout>
      <PageHeader
        label="Datapoint"
        name={datapoint.id}
        tag={
          <>
            {datapoint.is_custom && (
              <TooltipProvider delayDuration={200}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="ml-2 cursor-help">
                      Custom
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    This datapoint is not based on a historical inference. It
                    was either edited or created manually.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            {datapoint.staled_at && (
              <TooltipProvider delayDuration={200}>
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
              </TooltipProvider>
            )}
          </>
        }
      />

      {(saveError || validationError) && (
        <div className="mt-2 rounded-md bg-red-100 px-4 py-3 text-red-800">
          <p className="font-medium">Error saving datapoint</p>
          <p>{validationError || saveError}</p>
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
            isDeleting={fetcher.state === "submitting" && !saveError}
            toggleEditing={toggleEditing}
            isEditing={isEditing}
            canSave={canSave}
            onSave={handleSave}
            onReset={handleReset}
            showTryWithButton={datapoint.function_name !== DEFAULT_FUNCTION}
            isStale={!!datapoint.staled_at}
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
