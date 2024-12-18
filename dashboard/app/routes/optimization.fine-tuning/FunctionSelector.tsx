import { Control } from "react-hook-form";
import { SFTFormValues } from "./types";
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
              onValueChange={(value) => {
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
                      <div className="flex items-center justify-between w-full">
                        <span>{name}</span>
                        <span
                          className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                                    ${
                                      fn.type === "chat"
                                        ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
                                        : "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
                                    }`}
                        >
                          {fn.type === "chat"
                            ? "Chat"
                            : fn.type === "json"
                              ? "JSON"
                              : "Unknown"}
                        </span>
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
