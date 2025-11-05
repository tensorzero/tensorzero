import { z } from "zod";
import type {
  TagFilter,
  TagComparisonOperator,
  FloatComparisonOperator,
} from "~/types/tensorzero";

// Tag Filter
const tagFilterComparisonOperators =
  enforceEnumTypeMatches<TagComparisonOperator>()(["=", "!="]);

const tagFilterSchema = z.object({
  type: z.literal("tag"),
  key: z.string().min(1, "Tag key is required"),
  value: z.string().min(1, "Tag value is required"),
  comparison_operator: z.enum(tagFilterComparisonOperators),
}) satisfies z.ZodType<{ type: "tag" } & TagFilter>;

// Float Metric Filter
const floatFilterComparisonOperators =
  enforceEnumTypeMatches<FloatComparisonOperator>()([
    "<",
    "<=",
    "=",
    ">",
    ">=",
    "!=",
  ]);

const floatMetricFilterSchema = z.object({
  type: z.literal("float_metric"),
  metric_name: z.string().min(1, "Metric name is required"),
  value: z.number().finite("Value must be a finite number"),
  comparison_operator: z.enum(floatFilterComparisonOperators),
});

// Boolean Metric Filter
const booleanMetricFilterSchema = z.object({
  type: z.literal("boolean_metric"),
  metric_name: z.string().min(1, "Metric name is required"),
  value: z.boolean(),
});

// Recursive inference filter schema
export const InferenceFilterSchema: z.ZodTypeAny = z.lazy(() =>
  z.union([
    z.discriminatedUnion("type", [
      tagFilterSchema,
      floatMetricFilterSchema,
      booleanMetricFilterSchema,
    ]),
    z.object({
      type: z.enum(["and", "or"]),
      children: z.array(InferenceFilterSchema),
    }),
  ]),
);

export type InferenceFilterSchemaType = z.infer<typeof InferenceFilterSchema>;

// ===== BIDIRECTIONAL TYPECHECKING =====

// Helper to enforce bidirectional type equality
type BidirectionalCheck<TUnion, TArray> = [TUnion] extends [TArray]
  ? [TArray] extends [TUnion]
    ? unknown
    : never
  : never;

// Factory function that creates bidirectionally type-checked const arrays
function enforceEnumTypeMatches<TRustType extends string>() {
  return <const TArray extends readonly TRustType[]>(
    array: TArray & BidirectionalCheck<TRustType, TArray[number]>,
  ): TArray => array;
}
