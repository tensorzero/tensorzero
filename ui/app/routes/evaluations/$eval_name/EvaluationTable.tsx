"use client";
import { useState } from "react";
import React from "react";

import { Info, Check, X } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Badge } from "~/components/ui/badge";

// Import the evaluation data from the separate file
// import { evaluationData, variants } from "@/data/evaluation-data";
// import {
//   VariantSelector,
//   getVariantColor,
//   getLastUuidSegment,
// } from "@/components/variant-selector";

// Import the custom tooltip styles
import "./tooltip-styles.css";

// Update the TruncatedText component to truncate earlier and never wrap
const TruncatedText = ({ text, maxLength = 30, noWrap = false }) => {
  const truncated =
    text.length > maxLength ? text.slice(0, maxLength) + "..." : text;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={`flex cursor-help items-center gap-1 ${noWrap ? "overflow-hidden text-ellipsis whitespace-nowrap" : ""}`}
          >
            <span className="font-mono text-sm">{truncated}</span>
            {text.length > maxLength && (
              <Info className="h-3 w-3 flex-shrink-0 text-muted-foreground" />
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent
          side="right"
          align="start"
          sideOffset={5}
          className="tooltip-scrollable max-h-[60vh] max-w-md overflow-auto p-4 shadow-lg"
          avoidCollisions={true}
        >
          <pre className="whitespace-pre-wrap text-xs">{text}</pre>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// Define evaluator properties - update LLM Judge 4 to have a cutoff of 10%
const evaluatorProperties = {
  exactMatch: { type: "boolean", goal: "max", cutoff: null },
  llmJudge1: { type: "boolean", goal: "max", cutoff: 0.9 },
  llmJudge2: { type: "float", goal: "max", cutoff: 0.8 },
  llmJudge3: { type: "float", goal: "max", cutoff: null },
  llmJudge4: { type: "boolean", goal: "min", cutoff: 0.1 }, // Added cutoff of 10%
  llmJudge5: { type: "int", goal: "min", cutoff: 25 },
};

// Component to display evaluator properties as pills - updated to be more subtle
const EvaluatorProperties = ({ properties }) => {
  return (
    <div className="mt-2 flex flex-col items-center gap-1">
      <div className="flex justify-center gap-1">
        <Badge
          variant="outline"
          className="border-muted bg-transparent text-xs text-muted-foreground"
        >
          {properties.type}
        </Badge>
        <Badge
          variant="outline"
          className="border-muted bg-transparent text-xs text-muted-foreground"
        >
          {properties.goal}
        </Badge>
      </div>
      {properties.cutoff !== null && (
        <div className="mt-1 flex justify-center">
          <Badge
            variant="outline"
            className="border-muted bg-transparent text-center text-xs text-muted-foreground"
          >
            cutoff: {properties.cutoff}
          </Badge>
        </div>
      )}
    </div>
  );
};

// Update the EvaluatorResult component to swap icons for boolean min
const EvaluatorResult = ({ value, evaluatorType, isSummary = false }) => {
  const properties = evaluatorProperties[evaluatorType];

  // Handle boolean values
  if (
    typeof value === "boolean" ||
    (isSummary && properties.type === "boolean")
  ) {
    // For summary row, value is a percentage (0-1)
    const displayValue = isSummary
      ? `${Math.round(value * 100)}%`
      : value
        ? "True"
        : "False";

    // Determine which icon to show based on goal (max or min)
    let icon = null;
    if (!isSummary) {
      if (properties.goal === "max") {
        // For max goal: checkmark for true, X for false
        icon = value ? (
          <Check className="mr-1 h-3 w-3 flex-shrink-0" />
        ) : (
          <X className="mr-1 h-3 w-3 flex-shrink-0" />
        );
      } else {
        // For min goal: X for true, checkmark for false (swapped)
        icon = value ? (
          <X className="mr-1 h-3 w-3 flex-shrink-0" />
        ) : (
          <Check className="mr-1 h-3 w-3 flex-shrink-0" />
        );
      }
    }

    // Boolean max: true/high % = green, false/low % = red
    if (properties.goal === "max") {
      // For summary row with cutoff
      if (isSummary && properties.cutoff !== null) {
        const textColor =
          value >= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {displayValue}
          </span>
        );
      }
      // Regular boolean or summary without cutoff
      else {
        return (
          <span
            className={`${value ? "text-green-700" : "text-red-700"} flex items-center whitespace-nowrap`}
          >
            {icon}
            {displayValue}
          </span>
        );
      }
    }
    // Boolean min: true/high % = red, false/low % = green
    else {
      // For summary row with cutoff
      if (isSummary && properties.cutoff !== null) {
        const textColor =
          value <= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {displayValue}
          </span>
        );
      }
      // Regular boolean or summary without cutoff
      else {
        return (
          <span
            className={`${value ? "text-red-700" : "text-green-700"} flex items-center whitespace-nowrap`}
          >
            {icon}
            {displayValue}
          </span>
        );
      }
    }
  }

  // Rest of the function remains unchanged
  // Handle numeric values
  else if (typeof value === "number") {
    // For LLM Judge 5 (int type) or any int in summary
    if (
      properties.type === "int" ||
      (isSummary && evaluatorType === "llmJudge5")
    ) {
      // Format the value - for summary, round to nearest integer
      const displayValue = isSummary ? Math.round(value) : value;

      // Min goal with cutoff
      if (properties.goal === "min" && properties.cutoff !== null) {
        const textColor =
          value <= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {displayValue}
          </span>
        );
      }
      // Max goal with cutoff
      else if (properties.goal === "max" && properties.cutoff !== null) {
        const textColor =
          value >= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {displayValue}
          </span>
        );
      }
      // No cutoff - use grayscale
      else {
        return (
          <span className="whitespace-nowrap text-gray-700">
            {displayValue}
          </span>
        );
      }
    }
    // For float values (0-1 range)
    else if (
      (value >= 0 && value <= 1) ||
      (isSummary &&
        (evaluatorType === "llmJudge2" || evaluatorType === "llmJudge3"))
    ) {
      const percentage = Math.round(value * 100);

      // Float max with cutoff
      if (properties.goal === "max" && properties.cutoff !== null) {
        const textColor =
          value >= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {percentage}%
          </span>
        );
      }
      // Float min with cutoff
      else if (properties.goal === "min" && properties.cutoff !== null) {
        const textColor =
          value <= properties.cutoff ? "text-green-700" : "text-red-700";

        return (
          <span className={`${textColor} whitespace-nowrap`}>
            {percentage}%
          </span>
        );
      }
      // No cutoff - use grayscale
      else {
        return (
          <span className="whitespace-nowrap text-gray-700">{percentage}%</span>
        );
      }
    }
  }

  return <span className="whitespace-nowrap">{value.toString()}</span>;
};

// Calculate summary values for the evaluators by variant
const calculateSummaryByVariant = (data, variantId) => {
  const summary = {
    exactMatch: 0,
    llmJudge1: 0,
    llmJudge2: 0,
    llmJudge3: 0,
    llmJudge4: 0,
    llmJudge5: 0,
  };

  let count = 0;

  // Count the number of true values for boolean evaluators
  // Calculate the sum for numeric evaluators
  data.forEach((row) => {
    const variantData = row.variants.find((v) => v.variantId === variantId);
    if (variantData) {
      count++;
      Object.keys(summary).forEach((key) => {
        const value = variantData.evaluators[key];
        if (typeof value === "boolean") {
          summary[key] += value ? 1 : 0;
        } else {
          summary[key] += value;
        }
      });
    }
  });

  // Convert counts to percentages for boolean evaluators
  // Calculate averages for numeric evaluators
  if (count > 0) {
    Object.keys(summary).forEach((key) => {
      const properties = evaluatorProperties[key];
      if (properties.type === "boolean") {
        summary[key] = summary[key] / count;
      } else {
        summary[key] = summary[key] / count;
      }
    });
  }

  return summary;
};

// Component for variant label with color coding and run ID tooltip
const VariantLabel = ({ variantId }) => {
  const variant = variants.find((v) => v.id === variantId);
  if (!variant) return null;

  const colorClass = getVariantColor(variantId);
  const runIdSegment = getLastUuidSegment(variant.runId);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            className={`${colorClass} flex cursor-help items-center gap-1.5 px-2 py-1`}
          >
            <span>{variant.name}</span>
            <span className="border-l border-white/30 pl-1.5 text-xs opacity-80">
              {runIdSegment}
            </span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-2">
          <p className="text-xs">Run ID: {variant.runId}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export function EvaluationTable() {
  // State for selected variants - start with only baseline selected
  const [selectedVariants, setSelectedVariants] = useState(["baseline"]);

  // Determine if we should show the variant column
  const showVariantColumn = selectedVariants.length > 1;

  // Calculate summary values for each selected variant
  const summaryByVariant = {};
  selectedVariants.forEach((variantId) => {
    summaryByVariant[variantId] = calculateSummaryByVariant(
      evaluationData,
      variantId,
    );
  });

  return (
    <div>
      {/* Variant selector */}
      <VariantSelector
        selectedVariants={selectedVariants}
        onVariantChange={setSelectedVariants}
      />

      <div className="overflow-x-auto">
        <div className="min-w-max">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="py-2 text-center">Input</TableHead>
                <TableHead className="py-2 text-center">
                  Reference Output
                </TableHead>
                {showVariantColumn && (
                  <TableHead className="text-center">Variant</TableHead>
                )}
                <TableHead className="py-2 text-center">
                  Generated Output
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>Exact Match</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.exactMatch}
                  />
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>LLM Judge 1</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.llmJudge1}
                  />
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>LLM Judge 2</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.llmJudge2}
                  />
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>LLM Judge 3</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.llmJudge3}
                  />
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>LLM Judge 4</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.llmJudge4}
                  />
                </TableHead>
                <TableHead className="py-2 text-center">
                  <div>LLM Judge 5</div>
                  <EvaluatorProperties
                    properties={evaluatorProperties.llmJudge5}
                  />
                </TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {/* Group rows by input */}
              {evaluationData.map((inputData) => {
                const filteredVariants = inputData.variants.filter((v) =>
                  selectedVariants.includes(v.variantId),
                );

                if (filteredVariants.length === 0) return null;

                return (
                  <React.Fragment key={inputData.inputId}>
                    {filteredVariants.map((variant, index) => (
                      <TableRow
                        key={`input-${inputData.inputId}-variant-${variant.variantId}`}
                      >
                        {/* Input cell - only for the first variant row */}
                        {index === 0 && (
                          <TableCell
                            rowSpan={filteredVariants.length}
                            className="max-w-[200px] align-top"
                          >
                            <TruncatedText text={inputData.input} />
                          </TableCell>
                        )}

                        {/* Reference Output cell - only for the first variant row */}
                        {index === 0 && (
                          <TableCell
                            rowSpan={filteredVariants.length}
                            className="max-w-[200px] align-top"
                          >
                            <TruncatedText text={inputData.referenceOutput} />
                          </TableCell>
                        )}

                        {/* Variant label - only if multiple variants are selected */}
                        {showVariantColumn && (
                          <TableCell className="text-center align-middle">
                            <VariantLabel variantId={variant.variantId} />
                          </TableCell>
                        )}

                        {/* Generated output */}
                        <TableCell className="max-w-[200px] align-middle">
                          <TruncatedText
                            text={variant.generatedOutput}
                            noWrap={true}
                          />
                        </TableCell>

                        {/* Evaluator results */}
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.exactMatch}
                              evaluatorType="exactMatch"
                            />
                          </div>
                        </TableCell>
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.llmJudge1}
                              evaluatorType="llmJudge1"
                            />
                          </div>
                        </TableCell>
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.llmJudge2}
                              evaluatorType="llmJudge2"
                            />
                          </div>
                        </TableCell>
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.llmJudge3}
                              evaluatorType="llmJudge3"
                            />
                          </div>
                        </TableCell>
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.llmJudge4}
                              evaluatorType="llmJudge4"
                            />
                          </div>
                        </TableCell>
                        <TableCell className="h-[52px] text-center align-middle">
                          <div className="flex h-full items-center justify-center">
                            <EvaluatorResult
                              value={variant.evaluators.llmJudge5}
                              evaluatorType="llmJudge5"
                            />
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </React.Fragment>
                );
              })}

              {/* Summary Row */}
              <TableRow className="bg-muted/50 font-medium">
                <TableCell colSpan={2} className="text-left">
                  Summary ({evaluationData.length} inputs)
                </TableCell>

                {/* If showing variant column, add variant badges */}
                {showVariantColumn ? (
                  <TableCell className="align-middle">
                    <div className="flex flex-col gap-2">
                      {selectedVariants.map((variantId) => (
                        <div
                          key={`summary-variant-${variantId}`}
                          className="flex justify-center"
                        >
                          <VariantLabel variantId={variantId} />
                        </div>
                      ))}
                    </div>
                  </TableCell>
                ) : null}

                {/* Empty cell for Generated Output column */}
                <TableCell />

                {/* Summary cells for each evaluator - without variant badges */}
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-exactMatch`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].exactMatch}
                        evaluatorType="exactMatch"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-llmJudge1`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].llmJudge1}
                        evaluatorType="llmJudge1"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-llmJudge2`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].llmJudge2}
                        evaluatorType="llmJudge2"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-llmJudge3`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].llmJudge3}
                        evaluatorType="llmJudge3"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-llmJudge4`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].llmJudge4}
                        evaluatorType="llmJudge4"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
                <TableCell className="text-center align-middle">
                  {selectedVariants.map((variantId) => (
                    <div
                      key={`summary-${variantId}-llmJudge5`}
                      className="flex justify-center py-1"
                    >
                      <EvaluatorResult
                        value={summaryByVariant[variantId].llmJudge5}
                        evaluatorType="llmJudge5"
                        isSummary={true}
                      />
                    </div>
                  ))}
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}
