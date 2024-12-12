import { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { FormValues } from "./route";
import { ModelOption } from "./model-options";

type ModelSelectorProps = {
  control: Control<FormValues>;
  models: ModelOption[];
};

function formatProvider(provider: string) {
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
                  (model) => model.name === value,
                );
                if (selectedModel) {
                  field.onChange(selectedModel);
                }
              }}
              defaultValue={field.value?.name}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.name} value={model.name}>
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
  );
}
