import { useForm, useWatch } from "react-hook-form";
import { Form, FormLabel } from "~/components/ui/form";
import {
  DatasetBuilderFormValuesResolver,
  type DatasetBuilderFormValues,
} from "./types";
import { FunctionFormField } from "~/components/function/FunctionFormField";
import { DatasetFormField } from "~/components/dataset/DatasetFormField";
import { useFunctionConfig } from "~/context/config";
import { useFetcher } from "react-router";
import { useEffect, useState } from "react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Badge } from "~/components/ui/badge";
import { VariantSelector } from "~/components/function/variant/VariantSelector";
import OutputSourceSelector from "./OutputSourceSelector";
import { logger } from "~/utils/logger";
import InferenceFilterBuilder from "~/components/querybuilder/InferenceFilterBuilder";
import type { InferenceFilter } from "~/types/tensorzero";

export function DatasetBuilderForm() {
  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "complete"
  >("idle");
  const [isNewDataset, setIsNewDataset] = useState<boolean | null>(null);

  // Local state for InferenceFilter (managed outside react-hook-form)
  const [inferenceFilter, setInferenceFilter] = useState<
    InferenceFilter | undefined
  >(undefined);

  const form = useForm<DatasetBuilderFormValues>({
    defaultValues: {
      dataset: "",
      type: "chat",
      function: undefined,
      variant_name: undefined,
      episode_id: undefined,
      search_query: undefined,
      filters: undefined,
      output_source: "none",
    },
    resolver: DatasetBuilderFormValuesResolver,
    mode: "onChange",
  });

  const { handleSubmit } = form;

  const formFetcher = useFetcher();

  const watchedFields = useWatch({
    control: form.control,
    name: ["function", "dataset"] as const,
  });

  const [functionName, selectedDataset] = watchedFields;
  const functionConfig = useFunctionConfig(functionName ?? "");

  // Update form type when function changes
  useEffect(() => {
    const functionType = functionConfig?.type;
    if (functionType) {
      form.setValue("type", functionType);
    }
  }, [functionName, functionConfig, form]);

  // Sync inferenceFilter state with form
  useEffect(() => {
    form.setValue("filters", inferenceFilter);
  }, [inferenceFilter, form]);

  // Handle form submission response
  useEffect(() => {
    if (formFetcher.data) {
      if (formFetcher.data.errors) {
        logger.error("Form submission error:", formFetcher.data.errors);
        setSubmissionPhase("idle");
        form.setError("root", {
          type: "submit",
          message:
            formFetcher.data.errors.message ||
            "An error occurred while processing your request",
        });
      } else if (formFetcher.data.success) {
        setSubmissionPhase("complete");
        form.clearErrors("root");
      }
    }
  }, [formFetcher.data, form]);

  // Form submission handler
  const onSubmit = async (data: DatasetBuilderFormValues) => {
    try {
      const submitData = new FormData();
      submitData.append("data", JSON.stringify(data));

      formFetcher.submit(submitData, { method: "POST" });
      setSubmissionPhase("submitting");
    } catch (error) {
      logger.error("Submission error:", error);
      setSubmissionPhase("idle");
    }
  };

  function getButtonText(isNewDataset: boolean | null) {
    switch (submissionPhase) {
      case "submitting":
        return "Creating Dataset...";
      case "complete":
        return "Success";
      case "idle":
      default:
        if (isNewDataset) {
          return "Create Dataset";
        } else {
          return "Insert Into Dataset";
        }
    }
  }

  return (
    <Form {...form}>
      <form
        onSubmit={(e) => {
          handleSubmit(onSubmit)(e);
        }}
        className="space-y-6"
      >
        <div className="space-y-6">
          <DatasetFormField
            control={form.control}
            name="dataset"
            label="Dataset"
            placeholder="Select a dataset"
            onSelect={(dataset, isNew) => {
              setIsNewDataset(isNew);
            }}
          />

          <FunctionFormField
            control={form.control}
            name="function"
            onSelect={() => {
              form.resetField("variant_name");
            }}
          />

          <div className="grid gap-x-8 md:grid-cols-2">
            <div className="border-border bg-muted/30 rounded-lg border p-4">
              <div className="mb-4 flex items-center gap-2">
                <h3 className="text-muted-foreground text-sm font-semibold">
                  Filters
                </h3>
                <Badge
                  variant="outline"
                  className="text-muted-foreground text-xs"
                >
                  Optional
                </Badge>
              </div>
              <div className="space-y-4">
                <div>
                  <FormLabel>Variant</FormLabel>
                  <div className="mt-2">
                    <VariantSelector
                      functionName={functionName ?? null}
                      value={form.watch("variant_name") ?? ""}
                      onChange={(value) =>
                        form.setValue(
                          "variant_name",
                          value === "__all__" ? undefined : value || undefined,
                          { shouldValidate: true },
                        )
                      }
                    />
                  </div>
                </div>

                <div>
                  <FormLabel>Episode ID</FormLabel>
                  <div className="mt-2">
                    <Input
                      value={form.watch("episode_id") ?? ""}
                      onChange={(e) =>
                        form.setValue(
                          "episode_id",
                          e.target.value || undefined,
                          {
                            shouldValidate: true,
                          },
                        )
                      }
                      placeholder="00000000-0000-0000-0000-000000000000"
                    />
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2">
                    <FormLabel>Search Query</FormLabel>
                    <Badge variant="outline" className="text-xs">
                      Experimental
                    </Badge>
                  </div>
                  <div className="mt-2">
                    <Input
                      value={form.watch("search_query") ?? ""}
                      onChange={(e) =>
                        form.setValue(
                          "search_query",
                          e.target.value || undefined,
                          {
                            shouldValidate: true,
                          },
                        )
                      }
                      placeholder="Search in input and output"
                    />
                  </div>
                </div>

                <div>
                  <InferenceFilterBuilder
                    inferenceFilter={inferenceFilter}
                    setInferenceFilter={setInferenceFilter}
                  />
                </div>
              </div>
            </div>
          </div>

          <OutputSourceSelector control={form.control} />
        </div>
        <Button
          type="submit"
          disabled={
            submissionPhase !== "idle" || !selectedDataset || !functionName
          }
          onClick={() => {
            if (submissionPhase === "complete") {
              setSubmissionPhase("idle");
              form.clearErrors("root");
            }
          }}
        >
          {getButtonText(isNewDataset)}
        </Button>
        {form.formState.errors.root && (
          <p className="mt-2 text-sm text-red-500">
            {form.formState.errors.root.message}
          </p>
        )}
      </form>
    </Form>
  );
}
