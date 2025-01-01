import type { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import type { SFTFormValues } from "./types";
import type { ModelOption } from "./model_options";
import type { ProviderConfigSchema } from "~/utils/config/models";
import { z } from "zod";

type ProviderType = z.infer<typeof ProviderConfigSchema>["type"];

type ModelSelectorProps = {
  control: Control<SFTFormValues>;
  models: ModelOption[];
};

function formatProvider(provider: ProviderType): {
  name: string;
  className: string;
} {
  switch (provider) {
    case "anthropic":
      return {
        name: "Anthropic",
        className:
          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
      };
    case "aws_bedrock":
      return {
        name: "AWS Bedrock",
        className:
          "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
      };
    case "azure":
      return {
        name: "Azure",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "dummy":
      return {
        name: "Dummy",
        className:
          "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300",
      };
    case "fireworks":
      return {
        name: "Fireworks",
        className:
          "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300",
      };
    case "gcp_vertex_anthropic":
      return {
        name: "GCP Vertex AI (Anthropic)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "gcp_vertex_gemini":
      return {
        name: "GCP Vertex AI (Gemini)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "google_ai_studio_gemini":
      return {
        name: "Google AI Studio",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "hyperbolic":
      return {
        name: "Hyperbolic",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "mistral":
      return {
        name: "Mistral",
        className:
          "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
      };
    case "openai":
      return {
        name: "OpenAI",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "together":
      return {
        name: "Together",
        className:
          "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300",
      };
    case "vllm":
      return {
        name: "vLLM",
        className:
          "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300",
      };
    case "xai":
      return {
        name: "xAI",
        className:
          "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300",
      };
  }
}

export function ModelSelector({ control, models }: ModelSelectorProps) {
  return (
    <FormField
      control={control}
      name="model"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Model</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Select
              onValueChange={(value: string) => {
                const selectedModel = models.find(
                  (model) => model.displayName === value,
                );
                if (selectedModel) {
                  field.onChange(selectedModel);
                }
              }}
              defaultValue={field.value?.displayName}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.displayName} value={model.displayName}>
                    <div className="flex w-full items-center justify-between">
                      <span>{model.displayName}</span>
                      <span
                        className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          formatProvider(model.provider).className
                        }`}
                      >
                        {formatProvider(model.provider).name}
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
  );
}
