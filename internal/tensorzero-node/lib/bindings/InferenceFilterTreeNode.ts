// This file was generated by [ts-rs](https://github.com/Aleph-Alpha/ts-rs). Do not edit this file manually.
import type { BooleanMetricNode } from "./BooleanMetricNode";
import type { FloatMetricNode } from "./FloatMetricNode";

export type InferenceFilterTreeNode =
  | ({ type: "float_metric" } & FloatMetricNode)
  | ({ type: "boolean_metric" } & BooleanMetricNode)
  | { type: "and"; children: Array<InferenceFilterTreeNode> }
  | { type: "or"; children: Array<InferenceFilterTreeNode> }
  | { type: "not"; child: InferenceFilterTreeNode };
