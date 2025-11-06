import type { Control } from "react-hook-form";
import type { SFTFormValues } from "./types";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import type { ChatCompletionConfig } from "~/types/tensorzero";
import { TemplateDetailsDialog } from "./TemplateDetailsDialog";

type VariantSelectorProps = {
  control: Control<SFTFormValues>;
  chatCompletionVariants: Record<string, ChatCompletionConfig>;
};

export function VariantSelector({
  control,
  chatCompletionVariants,
}: VariantSelectorProps) {
  const hasVariants = Object.keys(chatCompletionVariants).length > 0;

  return (
    <FormField
      control={control}
      name="variant"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Prompt</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Select
              onValueChange={field.onChange}
              defaultValue={field.value}
              disabled={!hasVariants}
            >
              <SelectTrigger>
                <SelectValue
                  placeholder={
                    hasVariants
                      ? "Select a variant name"
                      : "No variants available"
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(chatCompletionVariants).map(([name]) => (
                  <SelectItem key={name} value={name}>
                    <span>{name}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <TemplateDetailsDialog
              variant={field.value}
              disabled={!field.value}
              chatCompletionVariants={chatCompletionVariants}
            />
          </div>
        </FormItem>
      )}
    />
  );
}
