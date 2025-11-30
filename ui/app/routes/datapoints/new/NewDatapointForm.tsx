import { useEffect, useState } from "react";
import { useFetcher } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { InputElement } from "~/components/input_output/InputElement";
import { Output } from "~/components/inference/Output";
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
} from "~/types/tensorzero";
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

  const functionConfig = useFunctionConfig(selectedFunction);
  const functionType = functionConfig?.type;

  // Reset output when function type changes
  useEffect(() => {
    if (functionType === "chat") {
      setOutput(DEFAULT_CHAT_OUTPUT);
    } else if (functionType === "json") {
      setOutput(DEFAULT_JSON_OUTPUT);
    } else {
      setOutput(undefined);
    }
  }, [functionType]);

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

    const formData = serializeCreateDatapointToFormData({
      dataset_name: selectedDataset,
      function_name: selectedFunction,
      function_type: functionType,
      input,
      output,
      tags: Object.keys(tags).length > 0 ? tags : undefined,
      name: name.trim() || undefined,
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
          <div className="space-y-2">
            <Label htmlFor="dataset">Dataset *</Label>
            <DatasetSelector
              selected={selectedDataset ?? undefined}
              onSelect={(dataset) => setSelectedDataset(dataset)}
              placeholder="Select dataset..."
              allowCreation
              disabled={isReadOnly}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="function">Function *</Label>
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
            {/* TODO (DO NOT MERGE): Migrate to the new components + allow editing the output schema. */}
            <SectionHeader heading="Output" />
            {output !== undefined && (
              <Output
                output={output}
                isEditing={true}
                onOutputChange={(newOutput) => setOutput(newOutput)}
              />
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
