import { useEffect, useState } from "react";
import { useFetcher } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import {
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { TagsTable } from "~/components/tags/TagsTable";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { useAllFunctionConfigs, useFunctionConfig } from "~/context/config";
import { useReadOnly } from "~/context/read-only";
import { useToast } from "~/hooks/use-toast";
import type {
  ContentBlockChatOutput,
  Input as InputType,
  JsonInferenceOutput,
  JsonValue,
} from "~/types/tensorzero";
import { validateJsonSchema } from "~/utils/jsonschema";
import { serializeCreateDatapointToFormData } from "./formDataUtils";

const DEFAULT_INPUT: InputType = {
  messages: [],
};

const DEFAULT_CHAT_OUTPUT: ContentBlockChatOutput[] = [];

const DEFAULT_JSON_OUTPUT: JsonInferenceOutput = {
  raw: "{}",
  parsed: {},
};

export function NewDatapointForm() {
  const functions = useAllFunctionConfigs();
  const isReadOnly = useReadOnly();
  const { toast } = useToast();
  const fetcher = useFetcher();

  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);
  const [input, setInput] = useState<InputType>(DEFAULT_INPUT);
  const [output, setOutput] = useState<
    ContentBlockChatOutput[] | JsonInferenceOutput | undefined
  >(undefined);
  const [tags, setTags] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [outputSchema, setOutputSchema] = useState<JsonValue | undefined>(
    undefined,
  );
  const [validationError, setValidationError] = useState<string | null>(null);

  const functionConfig = useFunctionConfig(selectedFunction);
  const functionType = functionConfig?.type;

  // Reset output and output schema when function config changes
  useEffect(() => {
    if (functionConfig?.type === "chat") {
      setOutput(DEFAULT_CHAT_OUTPUT);
      setOutputSchema(undefined);
    } else if (functionConfig?.type === "json") {
      setOutput(DEFAULT_JSON_OUTPUT);
      setOutputSchema(functionConfig.output_schema.value);
    } else {
      setOutput(undefined);
      setOutputSchema(undefined);
    }
    setValidationError(null);
  }, [functionConfig]);

  // Handle form submission errors
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      if (fetcher.data.success === false && fetcher.data.error) {
        toast.error({
          title: "Failed to create datapoint",
          description: fetcher.data.error,
        });
      }
    }
  }, [fetcher.state, fetcher.data, toast]);

  const handleSubmit = () => {
    if (!selectedDataset || !selectedFunction || !functionType) {
      toast.error({
        title: "Validation Error",
        description: "Please select a dataset and function.",
      });
      return;
    }

    // Validate output schema for JSON functions
    if (functionType === "json" && outputSchema !== undefined) {
      const schemaValidation = validateJsonSchema(outputSchema);
      if (!schemaValidation.valid) {
        setValidationError(schemaValidation.error);
        return;
      }
    }

    setValidationError(null);

    // Only include output_schema if it differs from the function's default
    const defaultSchema =
      functionConfig?.type === "json"
        ? functionConfig.output_schema.value
        : undefined;
    const schemaModified =
      functionType === "json" &&
      JSON.stringify(outputSchema) !== JSON.stringify(defaultSchema);

    const formData = serializeCreateDatapointToFormData({
      dataset_name: selectedDataset,
      function_name: selectedFunction,
      function_type: functionType,
      input,
      output,
      tags: Object.keys(tags).length > 0 ? tags : undefined,
      name: name.trim() || undefined,
      output_schema: schemaModified ? outputSchema : undefined,
    });

    fetcher.submit(formData, { method: "post" });
  };

  const isSubmitting = fetcher.state === "submitting";
  const canSubmit =
    selectedDataset &&
    selectedFunction &&
    functionType &&
    !isSubmitting &&
    !isReadOnly;

  return (
    <SectionsGroup>
      <SectionLayout>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2" data-testid="dataset-selector">
            <Label htmlFor="dataset">Dataset</Label>
            <DatasetSelector
              selected={selectedDataset ?? undefined}
              onSelect={(dataset) => setSelectedDataset(dataset)}
              allowCreation
              disabled={isReadOnly}
            />
          </div>
          <div className="space-y-2" data-testid="function-selector">
            <Label htmlFor="function">Function</Label>
            <FunctionSelector
              selected={selectedFunction}
              onSelect={setSelectedFunction}
              functions={functions}
              hideDefaultFunction
            />
          </div>
        </div>
      </SectionLayout>

      {selectedFunction && functionType && (
        <>
          <SectionLayout>
            <SectionHeader heading="Input" />
            <InputElement
              input={input}
              isEditing={true}
              onSystemChange={(system) => setInput({ ...input, system })}
              onMessagesChange={(messages) => setInput({ ...input, messages })}
            />
          </SectionLayout>
          <SectionLayout>
            <SectionHeader heading="Output" />
            {functionType === "json" ? (
              <JsonOutputElement
                output={output as JsonInferenceOutput | undefined}
                outputSchema={outputSchema}
                isEditing={true}
                onOutputChange={(newOutput) => {
                  setOutput(newOutput);
                  setValidationError(null);
                }}
                onOutputSchemaChange={(schema) => {
                  setOutputSchema(schema);
                  setValidationError(null);
                }}
              />
            ) : (
              <ChatOutputElement
                output={output as ContentBlockChatOutput[] | undefined}
                isEditing={true}
                onOutputChange={(newOutput) => setOutput(newOutput)}
              />
            )}
            {validationError && (
              <div className="mt-2 text-sm text-red-600">{validationError}</div>
            )}
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Tags" />
            <TagsTable tags={tags} onTagsChange={setTags} isEditing={true} />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Metadata" />
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter a name (optional)"
                disabled={isReadOnly}
              />
            </div>
          </SectionLayout>

          <SectionLayout>
            <div className="flex justify-end">
              <Button onClick={handleSubmit} disabled={!canSubmit}>
                {isSubmitting ? "Creating..." : "Create Datapoint"}
              </Button>
            </div>
          </SectionLayout>
        </>
      )}
    </SectionsGroup>
  );
}
