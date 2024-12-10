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
import { ModelOption } from "./mock-data";

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
            <Select onValueChange={field.onChange} defaultValue={field.value}>
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
  );
}
