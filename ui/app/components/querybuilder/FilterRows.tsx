import { useState, useId, useEffect, memo } from "react";
import { z } from "zod";
import type {
  InferenceFilter,
  DatapointFilter,
  MetricConfig,
  TagComparisonOperator,
  FloatComparisonOperator,
} from "~/types/tensorzero";
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
import { DeleteButton } from "../ui/DeleteButton";

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

// Selector Components

interface ComparisonOperatorSelectProps<T extends string> {
  value: T;
  onChange: (value: T) => void;
  operators: Record<T, string>;
  ariaLabel: string;
}

const ComparisonOperatorSelect = memo(function ComparisonOperatorSelect<
  T extends string,
>({ value, onChange, operators, ariaLabel }: ComparisonOperatorSelectProps<T>) {
  return (
    <div className="w-14">
      <Select onValueChange={(val) => onChange(val as T)} value={value}>
        <SelectTrigger aria-label={ariaLabel}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(operators) as T[]).map((op) => (
            <SelectItem key={op} value={op}>
              {operators[op]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}) as <T extends string>(
  props: ComparisonOperatorSelectProps<T>,
) => React.ReactElement;

interface BooleanValueSelectProps {
  value: boolean;
  onChange: (value: boolean) => void;
}

const BooleanValueSelect = memo(function BooleanValueSelect({
  value,
  onChange,
}: BooleanValueSelectProps) {
  return (
    <div className="w-48">
      <Select
        onValueChange={(val) => onChange(val === "true")}
        value={value.toString()}
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
  );
});

// Row Components

export interface TagFilterRowProps<
  T extends InferenceFilter | DatapointFilter = InferenceFilter,
> {
  filter: T & { type: "tag" };
  onChange: (newFilter: T | undefined) => void;
}

export const TagFilterRow = memo(function TagFilterRow<
  T extends InferenceFilter | DatapointFilter,
>({ filter, onChange }: TagFilterRowProps<T>) {
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

        <ComparisonOperatorSelect
          value={filter.comparison_operator}
          onChange={(op) => onChange({ ...filter, comparison_operator: op })}
          operators={TAG_OPERATOR_LABELS}
          ariaLabel="Tag comparison operator"
        />

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
          label="Delete tag filter"
          icon="x"
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

        <ComparisonOperatorSelect
          value={filter.comparison_operator}
          onChange={(op) => onChange({ ...filter, comparison_operator: op })}
          operators={FLOAT_OPERATOR_LABELS}
          ariaLabel="Metric comparison operator"
        />

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
          label="Delete metric filter"
          icon="x"
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

        <BooleanValueSelect
          value={filter.value}
          onChange={(val) => onChange({ ...filter, value: val })}
        />

        <DeleteButton
          onDelete={() => onChange(undefined)}
          label="Delete metric filter"
          icon="x"
        />
      </div>
    </div>
  );
});
