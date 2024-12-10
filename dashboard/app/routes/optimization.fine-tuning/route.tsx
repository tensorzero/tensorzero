import {
  json,
  type LoaderFunctionArgs,
  type MetaFunction,
} from "@remix-run/node";
import { Form } from "~/components/ui/form";
import { Button } from "~/components/ui/button";
import { models } from "./mock-data";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { useForm } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
// import { functions, metrics, models, promptTemplates } from "./mock-data";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Skeleton } from "~/components/ui/skeleton";
import { Input } from "~/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Textarea } from "~/components/ui/textarea";
import { promptTemplateDetails } from "./mock-data";
import { useEffect, useMemo, useState } from "react";
import { useConfig } from "~/context/config";
import {
  countFeedbacksForMetric,
  countInferencesForFunction,
  getCuratedInferences,
} from "~/utils/clickhouse";
import { getConfig } from "~/utils/config.server";
import { useLoaderData, useSearchParams } from "@remix-run/react";
import { ChatCompletionConfig, get_template_env } from "~/utils/config/variant";
// import {
// create_fine_tuning_job,
// poll_fine_tuning_job,
// tensorzero_inference_to_openai_messages,
// upload_examples_to_openai,
// } from "~/utils/fine_tuning/openai";
import OpenAI from "openai";
export const meta: MetaFunction = () => {
  return [
    { title: "TensorZeroFine-Tuning Dashboard" },
    { name: "description", content: "Fine Tuning Optimization Dashboard" },
  ];
};

type FormValues = {
  function: string;
  metric: string;
  model: string;
  variant: string;
  validationSplit: number;
  maxSamples: number;
  threshold?: number;
};

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function");
  const metricName = url.searchParams.get("metric");

  let inferenceCount = null;
  let feedbackCount = null;
  let curatedInferences = null;
  const config = await getConfig();
  if (functionName) {
    inferenceCount = await countInferencesForFunction(
      functionName,
      config.functions[functionName],
    );
  }
  if (metricName) {
    feedbackCount = await countFeedbacksForMetric(
      metricName,
      config.metrics[metricName],
    );
  }
  if (functionName && metricName) {
    curatedInferences = await getCuratedInferences(
      functionName,
      config.functions[functionName],
      metricName,
      config.metrics[metricName],
    );
  }
  return json({ inferenceCount, feedbackCount, curatedInferences });
}

export default function FineTuning() {
  const { inferenceCount, feedbackCount, curatedInferences } =
    useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();

  const config = useConfig();
  const form = useForm<FormValues>({
    defaultValues: {
      validationSplit: 20,
      maxSamples: 100000,
      threshold: 0.5,
    },
  });

  const [submissionResult, setSubmissionResult] = useState<string | null>(null);
  const [, setCounter] = useState<number | null>(null);
  const [finalResult, setFinalResult] = useState<string | null>(null);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [submissionPhase, setSubmissionPhase] = useState<
    "idle" | "submitting" | "pending" | "complete"
  >("idle");

  const handleFunctionChange = (value: string) => {
    setSearchParams(
      (prev) => {
        if (value) {
          prev.set("function", value);
        } else {
          prev.delete("function");
        }
        return prev;
      },
      { replace: true },
    );
  };

  const handleMetricChange = (value: string) => {
    setSearchParams(
      (prev) => {
        prev.set("metric", value);
        return prev;
      },
      { replace: true },
    );
  };

  const getChatCompletionVariantsForFunction = useMemo((): Record<
    string,
    ChatCompletionConfig
  > => {
    const selectedFunction = searchParams.get("function");

    if (!selectedFunction || !config?.functions[selectedFunction]) {
      return {};
    }

    const functionConfig = config.functions[selectedFunction];
    return Object.fromEntries(
      Object.entries(functionConfig.variants || {}).filter(
        (entry): entry is [string, ChatCompletionConfig] =>
          entry[1].type === "chat_completion",
      ),
    );
  }, [config, searchParams]);

  useEffect(() => {
    if (inferenceCount !== null) {
      form.setValue("maxSamples", Math.min(100000, inferenceCount));
    }
  }, [inferenceCount, form]);

  async function onSubmit(data: FormValues) {
    try {
      setIsSubmitted(true);
      setSubmissionPhase("submitting");
      setSubmissionResult("Preparing training data...");

      const current_variant =
        getChatCompletionVariantsForFunction[data.variant];
      // const template_env = await get_template_env(current_variant);
      // const messages = curatedInferences?.map((inference) =>
      //   tensorzero_inference_to_openai_messages(inference, template_env)
      // );
      // if (!messages) {
      //   throw new Error("No messages found");
      // }

      setSubmissionResult("Uploading training data to OpenAI...");
      // const file_id = await upload_examples_to_openai(messages);
    } catch (err) {
      const error = err as Error;
      setSubmissionPhase("complete");
      setFinalResult(`Error during fine-tuning: ${error.message}`);
    }

    //   setSubmissionResult(
    //     `Training data uploaded (File ID: ${file_id})\nStarting fine-tuning job...`
    //   );
    //   const job_id = await create_fine_tuning_job(data.model, file_id);

    //   setSubmissionPhase("pending");
    //   let finished = false;
    //   let job: OpenAI.FineTuning.FineTuningJob | undefined;
    //   let counter = 1;

    //   while (!finished) {
    //     await new Promise((resolve) => setTimeout(resolve, 10000));
    //     job = await poll_fine_tuning_job(job_id);

    //     // Update UI with current status
    //     counter++;
    //     setCounter(counter);
    //     setSubmissionResult(
    //       `Attempt ${counter}\n\nFine-tuning job status: ${job.status}\n` +
    //         `Training progress: ${job.trained_tokens ?? 0} tokens\n` +
    //         `${job.status === "running" ? "Training in progress..." : ""}`
    //     );

    //     finished =
    //       job.status === "succeeded" ||
    //       job.status === "failed" ||
    //       job.status === "cancelled";
    //   }
    //   if (!job) {
    //     throw new Error("No job found after fine-tuning");
    //   }

    //   setSubmissionPhase("complete");
    //   setFinalResult(
    //     job.status === "succeeded"
    //       ? `Fine-tuning completed successfully!\n\n` +
    //           `Model ID: ${job.fine_tuned_model}\n` +
    //           `Training tokens: ${job.trained_tokens}\n` +
    //           `Training file: ${job.training_file}\n` +
    //           `Validation file: ${job.validation_file ?? "None"}`
    //       : `Fine-tuning failed with status: ${job.status}\n` +
    //           `${job.error?.message ?? "No error message provided"}`
    //   );
    // } catch (err) {
    //   const error = err as Error;
    //   setSubmissionPhase("complete");
    //   setFinalResult(`Error during fine-tuning: ${error.message}`);
    // }
  }

  // Helper function to format provider name
  function formatProvider(provider: string): string {
    switch (provider) {
      case "openai":
        return "OpenAI";
      case "anthropic":
        return "Anthropic";
      case "mistral":
        return "Mistral";
      default:
        return provider;
    }
  }

  function getButtonText() {
    switch (submissionPhase) {
      case "submitting":
        return "Submitting...";
      case "pending":
        return "Pending...";
      case "complete":
        return "Complete";
      default:
        return "Start Fine-tuning Job";
    }
  }
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="p-4">
        <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">
          Fine-Tuning
        </h2>

        <div className="mt-8">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <div className="space-y-6">
                <FormField
                  control={form.control}
                  name="function"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Function</FormLabel>
                      <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                        <Select
                          onValueChange={(value) => {
                            field.onChange(value);
                            handleFunctionChange(value);
                          }}
                          value={field.value}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select a function" />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(config.functions).map(
                              ([name, fn]) => {
                                return (
                                  <SelectItem key={name} value={name}>
                                    <div className="flex items-center justify-between w-full">
                                      <span>{name}</span>
                                      <span
                                        className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                    ${
                                      fn.type === "chat"
                                        ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
                                        : "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
                                    }`}
                                      >
                                        {fn.type === "chat"
                                          ? "Chat"
                                          : fn.type === "json"
                                            ? "JSON"
                                            : "Unknown"}
                                      </span>
                                    </div>
                                  </SelectItem>
                                );
                              },
                            )}
                          </SelectContent>
                        </Select>
                        <div className="text-sm text-muted-foreground">
                          Inferences:{" "}
                          {field.value ? (
                            <span className="font-medium">
                              {inferenceCount ?? (
                                <Skeleton className="inline-block h-4 w-16 align-middle" />
                              )}
                            </span>
                          ) : (
                            <Skeleton className="inline-block h-4 w-16 align-middle" />
                          )}
                        </div>
                      </div>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="metric"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Metric</FormLabel>
                      <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                        <div className="space-y-2">
                          <Select
                            onValueChange={(value) => {
                              field.onChange(value);
                              handleMetricChange(value);
                            }}
                            defaultValue={field.value}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select a metric" />
                            </SelectTrigger>
                            <SelectContent>
                              {Object.entries(config.metrics).map(
                                ([name, metric]) => (
                                  <SelectItem key={name} value={name}>
                                    <div className="flex items-center justify-between w-full">
                                      <span>{name}</span>
                                      <div className="ml-2 flex gap-1.5">
                                        <span
                                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                        ${
                                          metric.type === "boolean"
                                            ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
                                            : "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300"
                                        }`}
                                        >
                                          {metric.type}
                                        </span>
                                        <span
                                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                        ${
                                          metric.optimize === "max"
                                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
                                            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
                                        }`}
                                        >
                                          {metric.optimize}
                                        </span>
                                        <span
                                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                        ${
                                          metric.level === "episode"
                                            ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300"
                                            : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-300"
                                        }`}
                                        >
                                          {metric.level}
                                        </span>
                                      </div>
                                    </div>
                                  </SelectItem>
                                ),
                              )}
                            </SelectContent>
                          </Select>

                          {field.value &&
                            config.metrics[field.value]?.type === "float" && (
                              <FormField
                                control={form.control}
                                name="threshold"
                                render={({ field: thresholdField }) => (
                                  <div className="p-4 bg-gray-100 rounded-lg">
                                    <FormLabel>Threshold</FormLabel>
                                    <Input
                                      type="number"
                                      step="0.01"
                                      min={0}
                                      max={1}
                                      {...thresholdField}
                                      className="bg-transparent border-none focus:ring-0"
                                      onChange={(e) =>
                                        thresholdField.onChange(
                                          Number(e.target.value),
                                        )
                                      }
                                    />
                                  </div>
                                )}
                              />
                            )}
                        </div>

                        <div className="space-y-1 text-sm text-muted-foreground">
                          <div>
                            Feedbacks:{" "}
                            {field.value ? (
                              <span className="font-medium">
                                {feedbackCount}
                              </span>
                            ) : (
                              <Skeleton className="inline-block h-4 w-16 align-middle" />
                            )}
                          </div>
                          <div>
                            Curated Inferences:{" "}
                            {field.value && form.watch("function") ? (
                              <span className="font-medium">
                                {curatedInferences?.length}
                              </span>
                            ) : (
                              <Skeleton className="inline-block h-4 w-16 align-middle" />
                            )}
                          </div>
                        </div>
                      </div>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="variant"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Prompt Template</FormLabel>
                      <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                        <Select
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select a prompt template" />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(
                              getChatCompletionVariantsForFunction,
                            ).map(([name]) => (
                              <SelectItem key={name} value={name}>
                                <span>{name}</span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <div className="flex">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="outline" disabled={!field.value}>
                                Details
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-[625px] p-0 overflow-hidden">
                              <div className="max-h-[90vh] overflow-y-auto p-6 rounded-lg">
                                <DialogHeader>
                                  <DialogTitle>Template Details</DialogTitle>
                                </DialogHeader>
                                <div className="grid gap-4 py-4">
                                  <div className="space-y-4">
                                    {field.value && (
                                      <>
                                        <div className="space-y-2">
                                          <h4 className="font-medium leading-none">
                                            System Template
                                          </h4>
                                          {getChatCompletionVariantsForFunction[
                                            field.value
                                          ]?.system_template ? (
                                            <Textarea
                                              readOnly
                                              value={
                                                getChatCompletionVariantsForFunction[
                                                  field.value
                                                ]?.system_template
                                              }
                                              className="h-[200px] resize-none"
                                            />
                                          ) : (
                                            <p className="text-sm text-muted-foreground">
                                              No system template.
                                            </p>
                                          )}
                                        </div>

                                        <div className="space-y-2">
                                          <h4 className="font-medium leading-none">
                                            User Template
                                          </h4>
                                          {getChatCompletionVariantsForFunction[
                                            field.value
                                          ]?.user_template ? (
                                            <Textarea
                                              readOnly
                                              value={
                                                getChatCompletionVariantsForFunction[
                                                  field.value
                                                ]?.user_template
                                              }
                                              className="h-[200px] resize-none"
                                            />
                                          ) : (
                                            <p className="text-sm text-muted-foreground">
                                              No user template.
                                            </p>
                                          )}
                                        </div>
                                      </>
                                    )}

                                    <div className="space-y-2">
                                      <h4 className="font-medium leading-none">
                                        Assistant Template
                                      </h4>
                                      {promptTemplateDetails.assistant ? (
                                        <Textarea
                                          readOnly
                                          value={
                                            promptTemplateDetails.assistant
                                          }
                                          className="h-[200px] resize-none"
                                        />
                                      ) : (
                                        <p className="text-sm text-muted-foreground">
                                          No assistant template.
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </DialogContent>
                          </Dialog>
                        </div>
                      </div>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="model"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Model</FormLabel>
                      <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                        <Select
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select a model" />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(models).map(([name, model]) => (
                              <SelectItem key={name} value={name}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{model.name}</span>
                                  <span className="ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300">
                                    {formatProvider(model.provider)}
                                  </span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <div></div>
                      </div>
                    </FormItem>
                  )}
                />

                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="advanced-parameters">
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center gap-2">
                        <span>Advanced Parameters</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-6 pt-3 px-3">
                        <FormField
                          control={form.control}
                          name="validationSplit"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Validation Split (%)</FormLabel>
                              <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                                <Input
                                  type="number"
                                  min={0}
                                  max={100}
                                  {...field}
                                  onChange={(e) =>
                                    field.onChange(Number(e.target.value))
                                  }
                                />
                                <div></div>
                              </div>
                            </FormItem>
                          )}
                        />

                        <FormField
                          control={form.control}
                          name="maxSamples"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Max. Samples</FormLabel>
                              <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                                <Input
                                  type="number"
                                  min={1}
                                  step={1}
                                  {...field}
                                  onChange={(e) =>
                                    field.onChange(Number(e.target.value))
                                  }
                                />
                                <div></div>
                              </div>
                            </FormItem>
                          )}
                        />
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </div>

              <div className="space-y-4">
                <Button
                  type="submit"
                  disabled={
                    !form.watch("function") ||
                    !form.watch("metric") ||
                    !form.watch("model") ||
                    !form.watch("variant") ||
                    form.formState.isSubmitting ||
                    isSubmitted
                  }
                >
                  {getButtonText()}
                </Button>

                {submissionResult && (
                  <div className="p-4 bg-gray-100 rounded-lg">
                    <div className="mb-2 font-medium">Job Status</div>
                    <Textarea
                      value={submissionResult}
                      className="w-full h-48 resize-none bg-transparent border-none focus:ring-0"
                      readOnly
                    />
                  </div>
                )}

                {finalResult && (
                  <div className="p-4 bg-gray-100 rounded-lg">
                    <div className="mb-2 font-medium">Configuration</div>
                    <Textarea
                      value={finalResult}
                      className="w-full h-48 resize-none bg-transparent border-none focus:ring-0"
                      readOnly
                    />
                  </div>
                )}
              </div>
            </form>
          </Form>
        </div>
      </main>
    </div>
  );
}
