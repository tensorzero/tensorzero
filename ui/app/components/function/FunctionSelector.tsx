import type { Control, Path } from "react-hook-form";
import { Config } from "~/utils/config";
import { FormField, FormItem } from "~/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { FunctionBadges } from "~/components/function/FunctionBadges";

type FunctionSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  config: Config;
  hide_default_function?: boolean;
};

export function FunctionSelector<T extends Record<string, unknown>>({
  control,
  name,
  config,
  hide_default_function = false,
}: FunctionSelectorProps<T>) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-col gap-1">
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
                if (hide_default_function && name === "tensorzero::default") {
                  return null;
                }
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
          {/* keeping temporary until full update
          <div className="text-muted-foreground text-sm">
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
          */}
        </FormItem>
      )}
    />
  );
}
