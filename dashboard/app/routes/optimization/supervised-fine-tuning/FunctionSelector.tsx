import type { Control } from "react-hook-form";
import type { SFTFormValues } from "./types";
import { Config } from "~/utils/config";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Skeleton } from "~/components/ui/skeleton";
import { FunctionBadges } from "~/components/function/FunctionBadges";

type FunctionSelectorProps = {
  control: Control<SFTFormValues>;
  inferenceCount: number | null;
  config: Config;
  onFunctionChange: (value: string) => void;
};

export function FunctionSelector({
  control,
  inferenceCount,
  config,
  onFunctionChange,
}: FunctionSelectorProps) {
  return (
    <FormField
      control={control}
      name="function"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Function</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Select
              onValueChange={(value: string) => {
                field.onChange(value);
                onFunctionChange(value);
              }}
              value={field.value}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a function" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(config.functions).map(([name, fn]) => {
                  return (
                    <SelectItem key={name} value={name}>
                      <div className="flex w-full items-center justify-between">
                        <span>{name}</span>
                        <div className="ml-2">
                          <FunctionBadges fn={fn} />
                        </div>
                      </div>
                    </SelectItem>
                  );
                })}
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
  );
}
