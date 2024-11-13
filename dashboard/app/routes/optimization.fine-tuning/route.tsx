import {
  json,
  type LoaderFunctionArgs,
  type MetaFunction,
} from "@remix-run/node";
import { Form } from "~/components/ui/form";
import { Button } from "~/components/ui/button";
import { promptTemplates, models } from "./mock-data";
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
import { useEffect, useState } from "react";
import { useConfig } from "~/context/config";
import { countInferencesForFunction } from "~/utils/clickhouse";
import { getConfig } from "~/utils/config.server";
import { useLoaderData, useNavigate } from "@remix-run/react";

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
  promptTemplate: string;
  validationSplit: number;
  maxSamples: number;
  threshold?: number;
};

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function");
  if (!functionName) {
    return json({ inferenceCount: null });
  }
  const config = await getConfig();
  const inferenceCount = await countInferencesForFunction(
    functionName,
    config.functions[functionName]
  );
  return json({ inferenceCount });
}

export default function FineTuning() {
  const { inferenceCount } = useLoaderData<typeof loader>();
  const navigate = useNavigate();

  const config = useConfig();
  console.log(config);
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

  useEffect(() => {
    const functionValue = form.watch("function");
    if (functionValue) {
      navigate(`?function=${functionValue}`, { replace: true });
    }
  }, [form.watch("function"), navigate]);

  useEffect(() => {
    if (inferenceCount !== null) {
      form.setValue("maxSamples", Math.min(100000, inferenceCount));
    }
  }, [inferenceCount, form]);

  async function onSubmit(data: FormValues) {
    console.log(data);
    setIsSubmitted(true);
    setSubmissionPhase("submitting");

    // Wait 3 seconds before starting counter
    await new Promise((resolve) => setTimeout(resolve, 3000));

    // Start counter
    setSubmissionPhase("pending");
    setCounter(1);
    setSubmissionResult(
      `1\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.`
    );

    // Counter interval
    const interval = setInterval(() => {
      setCounter((prev) => {
        if (prev === 10) {
          clearInterval(interval);
          setSubmissionPhase("complete");
          setFinalResult(
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
          );
          return prev;
        }
        const newCount = prev! + 1;
        setSubmissionResult((prev) =>
          prev ? prev.replace(/^\d+/, newCount.toString()) : prev
        );
        return newCount;
      });
    }, 1000);
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
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select a function" />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(config.functions).map(
                              ([name, fn]) => {
                                console.log(name);
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
                              }
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
                            onValueChange={field.onChange}
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
                                )
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
                                          Number(e.target.value)
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
                              <span className="font-medium">123,456</span>
                            ) : (
                              <Skeleton className="inline-block h-4 w-16 align-middle" />
                            )}
                          </div>
                          <div>
                            Curated Inferences:{" "}
                            {field.value && form.watch("function") ? (
                              <span className="font-medium">12,345</span>
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
                  name="promptTemplate"
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
                            {Object.entries(promptTemplates).map(([name]) => (
                              <SelectItem key={name} value={name}>
                                <span>{name}</span>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <div className="flex">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="outline">Details</Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-[625px] p-0 overflow-hidden">
                              <div className="max-h-[90vh] overflow-y-auto p-6 rounded-lg">
                                <DialogHeader>
                                  <DialogTitle>Template Details</DialogTitle>
                                </DialogHeader>
                                <div className="grid gap-4 py-4">
                                  <div className="space-y-4">
                                    <div className="space-y-2">
                                      <h4 className="font-medium leading-none">
                                        System Template
                                      </h4>
                                      {promptTemplateDetails.system ? (
                                        <Textarea
                                          readOnly
                                          value={promptTemplateDetails.system}
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
                                      {promptTemplateDetails.user ? (
                                        <Textarea
                                          readOnly
                                          value={promptTemplateDetails.user}
                                          className="h-[200px] resize-none"
                                        />
                                      ) : (
                                        <p className="text-sm text-muted-foreground">
                                          No user template.
                                        </p>
                                      )}
                                    </div>

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
                    !form.watch("promptTemplate") ||
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
