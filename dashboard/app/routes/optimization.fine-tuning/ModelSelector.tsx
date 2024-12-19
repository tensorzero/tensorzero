import { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { SFTFormValues } from "./types";
import { ModelOption } from "./model_options";

type ModelSelectorProps = {
  control: Control<SFTFormValues>;
  models: ModelOption[];
};

function formatProvider(provider: string): { name: string; className: string } {
  switch (provider) {
    case "openai":
      return {
        name: "OpenAI",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "anthropic":
      return {
        name: "Anthropic",
        className:
          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
      };
    case "mistral":
      return {
        name: "Mistral",
        className:
          "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
      };
    default:
      return {
        name: provider,
        className:
          "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300",
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
              onValueChange={(value) => {
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
                    <div className="flex items-center justify-between w-full">
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
