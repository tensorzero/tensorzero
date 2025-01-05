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
import { ModelBadges } from "~/components/model/ModelBadges";

type ModelSelectorProps = {
  control: Control<SFTFormValues>;
  models: ModelOption[];
};

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
                      <div className="ml-2">
                        <ModelBadges provider={model.provider} />
                      </div>
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
