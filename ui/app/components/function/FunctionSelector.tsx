import type { Control, Path } from "react-hook-form";
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

type FunctionSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  inferenceCount: number | null;
  config: Config;
};

export function FunctionSelector<T extends Record<string, unknown>>({
  control,
  name,
  inferenceCount,
  config,
}: FunctionSelectorProps<T>) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>Function</FormLabel>
          <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
            <Select
              onValueChange={(value: string) => {
                field.onChange(value);
              }}
              value={field.value as string}
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
