import { useState, useId, useEffect, memo } from "react";
import { z } from "zod";
import type {
  InferenceFilter,
  MetricConfig,
  TagComparisonOperator,
  FloatComparisonOperator,
} from "tensorzero-node";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { cn } from "~/utils/common";
import { MetricNameWithTooltip } from "~/components/querybuilder/MetricNameWithTooltip";
import DeleteButton from "~/components/querybuilder/DeleteButton";

// Labels for comparison operators
export const TAG_OPERATOR_LABELS = {
  "=": "=",
  "!=": "≠",
} as const satisfies Record<TagComparisonOperator, string>;

export const FLOAT_OPERATOR_LABELS = {
  "<": "<",
  "<=": "≤",
  "=": "=",
  ">": ">",
  ">=": "≥",
  "!=": "≠",
} as const satisfies Record<FloatComparisonOperator, string>;

// Field validation
export const InferenceFilterFieldValidation = {
  floatValue: z
    .string()
    .min(1, "Required")
    .refine((val) => !isNaN(parseFloat(val)), "Must be a number")
    .refine((val) => isFinite(parseFloat(val)), "Must be a number"),
  tagKey: z.string().min(1, "Required"),
  tagValue: z.string().min(1, "Required"),
};

// Shared ValidatedInput component
interface ValidatedInputProps {
  value: string;
  onChange: (value: string) => void;
  onBlur: () => void;
  error?: string;
  placeholder?: string;
  ariaLabel: string;
  className?: string;
  type?: "text" | "number";
  step?: string;
}

const ValidatedInput = memo(function ValidatedInput({
  value,
  onChange,
  onBlur,
  error,
  placeholder,
  ariaLabel,
  className,
  type = "text",
  step,
}: ValidatedInputProps) {
  const errorId = useId();

  return (
    <div className="relative">
      <Input
        type={type}
        step={step}
        className={cn(className, error && "border-red-500")}
        placeholder={placeholder}
        aria-label={ariaLabel}
        aria-invalid={!!error}
        aria-describedby={error ? errorId : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
      />
      {error && (
        <p
          id={errorId}
          role="alert"
          className="text-destructive absolute top-1 right-2 text-xs"
        >
          {error}
        </p>
      )}
    </div>
  );
});

// Row Components

export interface TagFilterRowProps {
  filter: InferenceFilter & { type: "tag" };
  onChange: (newFilter: InferenceFilter | undefined) => void;
}

export const TagFilterRow = memo(function TagFilterRow({
  filter,
  onChange,
}: TagFilterRowProps) {
  const [keyError, setKeyError] = useState<string>();
  const [valueError, setValueError] = useState<string>();

  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <ValidatedInput
            value={filter.key}
            onChange={(val) => {
              onChange({ ...filter, key: val });
              setKeyError(undefined);
            }}
            onBlur={() => {
              const result = InferenceFilterFieldValidation.tagKey.safeParse(
                filter.key,
              );
              if (!result.success) {
                setKeyError(result.error.errors[0]?.message ?? "Invalid input");
              }
            }}
            error={keyError}
            placeholder="tag"
            ariaLabel="Tag key"
            className="font-mono"
          />
        </div>

        <div className="w-14">
          <Select
            onValueChange={(value) => {
              onChange({
                ...filter,
                comparison_operator: value as TagComparisonOperator,
              });
            }}
            value={filter.comparison_operator}
          >
            <SelectTrigger aria-label="Tag comparison operator">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(
                Object.keys(TAG_OPERATOR_LABELS) as TagComparisonOperator[]
              ).map((op) => (
                <SelectItem key={op} value={op}>
                  {TAG_OPERATOR_LABELS[op]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="relative w-48">
          <ValidatedInput
            value={filter.value}
            onChange={(val) => {
              onChange({ ...filter, value: val });
              setValueError(undefined);
            }}
            onBlur={() => {
              const result = InferenceFilterFieldValidation.tagValue.safeParse(
                filter.value,
              );
              if (!result.success) {
                setValueError(
                  result.error.errors[0]?.message ?? "Invalid input",
                );
              }
            }}
            error={valueError}
            placeholder="value"
            ariaLabel="Tag value"
            className="font-mono"
          />
        </div>

        <DeleteButton
          onDelete={() => onChange(undefined)}
          ariaLabel="Delete tag filter"
        />
      </div>
    </div>
  );
});

export interface FloatMetricFilterRowProps {
  filter: InferenceFilter & { type: "float_metric" };
  metricConfig: MetricConfig;
  onChange: (newFilter: InferenceFilter | undefined) => void;
}

export const FloatMetricFilterRow = memo(function FloatMetricFilterRow({
  filter,
  metricConfig,
  onChange,
}: FloatMetricFilterRowProps) {
  const [inputValue, setInputValue] = useState(filter.value.toString());
  const [error, setError] = useState<string>();

  // Sync with external changes to filter.value
  useEffect(() => {
    setInputValue(filter.value.toString());
  }, [filter.value]);

  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <MetricNameWithTooltip
            metricName={filter.metric_name}
            metricConfig={metricConfig}
          />
        </div>

        <div className="w-14">
          <Select
            onValueChange={(value) => {
              onChange({
                ...filter,
                comparison_operator: value as FloatComparisonOperator,
              });
            }}
            value={filter.comparison_operator}
          >
            <SelectTrigger aria-label="Metric comparison operator">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(
                Object.keys(FLOAT_OPERATOR_LABELS) as FloatComparisonOperator[]
              ).map((op) => (
                <SelectItem key={op} value={op}>
                  {FLOAT_OPERATOR_LABELS[op]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="relative w-48">
          <ValidatedInput
            type="number"
            step="any"
            value={inputValue}
            onChange={(val) => {
              setInputValue(val);
              setError(undefined);

              const parsed = parseFloat(val);
              if (!isNaN(parsed)) {
                onChange({ ...filter, value: parsed });
              }
            }}
            onBlur={() => {
              const result =
                InferenceFilterFieldValidation.floatValue.safeParse(inputValue);
              if (!result.success) {
                setError(result.error.errors[0]?.message ?? "Invalid input");
              }
            }}
            error={error}
            placeholder="0"
            ariaLabel="Metric threshold value"
          />
        </div>

        <DeleteButton
          onDelete={() => onChange(undefined)}
          ariaLabel="Delete metric filter"
        />
      </div>
    </div>
  );
});

export interface BooleanMetricFilterRowProps {
  filter: InferenceFilter & { type: "boolean_metric" };
  metricConfig: MetricConfig;
  onChange: (newFilter: InferenceFilter | undefined) => void;
}

export const BooleanMetricFilterRow = memo(function BooleanMetricFilterRow({
  filter,
  metricConfig,
  onChange,
}: BooleanMetricFilterRowProps) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <MetricNameWithTooltip
            metricName={filter.metric_name}
            metricConfig={metricConfig}
          />
        </div>

        <div className="text-fg-secondary flex w-14 items-center justify-center text-sm">
          is
        </div>

        <div className="w-48">
          <Select
            onValueChange={(value) =>
              onChange({ ...filter, value: value === "true" })
            }
            value={filter.value.toString()}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true">true</SelectItem>
              <SelectItem value="false">false</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <DeleteButton
          onDelete={() => onChange(undefined)}
          ariaLabel="Delete metric filter"
        />
      </div>
    </div>
  );
});
