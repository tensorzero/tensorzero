import type { Control } from "react-hook-form";
import type { SFTFormValues } from "./types";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { Combobox } from "~/components/ui/combobox";
import type { ChatCompletionConfig } from "~/types/tensorzero";
import { TemplateDetailsDialog } from "./TemplateDetailsDialog";
import { useMemo } from "react";

type VariantSelectorProps = {
  control: Control<SFTFormValues>;
  chatCompletionVariants: Record<string, ChatCompletionConfig>;
};

export function VariantSelector({
  control,
  chatCompletionVariants,
}: VariantSelectorProps) {
  const variantNames = useMemo(
    () => Object.keys(chatCompletionVariants),
    [chatCompletionVariants],
  );
  const hasVariants = variantNames.length > 0;

  return (
    <FormField
      control={control}
      name="variant"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Prompt</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Combobox
              selected={field.value || null}
              onSelect={(value) => field.onChange(value)}
              items={variantNames}
              placeholder={
                hasVariants ? "Select variant" : "No variants available"
              }
              emptyMessage="No variants found"
              disabled={!hasVariants}
              ariaLabel="Prompt"
            />
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
