"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { variants } from "@/data/evaluation-data";

interface VariantSelectorProps {
  selectedVariants: string[];
  onVariantChange: (variants: string[]) => void;
}

// Helper function to get the last 6 digits of a UUID
export function getLastUuidSegment(uuid: string): string {
  return uuid.slice(-6);
}

export function VariantSelector({
  selectedVariants,
  onVariantChange,
}: VariantSelectorProps) {
  // State to track if dropdown is open
  const [isOpen, setIsOpen] = useState(false);

  // Toggle a variant selection
  const toggleVariant = (variantId: string) => {
    if (selectedVariants.includes(variantId)) {
      // If all variants would be deselected, don't allow it
      if (selectedVariants.length === 1) return;

      onVariantChange(selectedVariants.filter((id) => id !== variantId));
    } else {
      onVariantChange([...selectedVariants, variantId]);
    }
  };

  // Select all variants
  const selectAll = () => {
    onVariantChange(variants.map((v) => v.id));
  };

  return (
    <div className="mb-6">
      <div className="flex flex-col space-y-2">
        <h2 className="text-lg font-semibold">Compare Variants</h2>

        <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className="flex w-56 items-center justify-between gap-2"
            >
              <span>Select run ID...</span>
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            <DropdownMenuLabel>Select Variants</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {variants.map((variant) => {
              const isSelected = selectedVariants.includes(variant.id);
              const variantColor = getVariantColorClass(variant.id, isSelected);
              const runIdSegment = getLastUuidSegment(variant.runId);

              return (
                <DropdownMenuCheckboxItem
                  key={variant.id}
                  checked={isSelected}
                  onCheckedChange={() => toggleVariant(variant.id)}
                  disabled={isSelected && selectedVariants.length === 1}
                  className="flex items-center gap-2"
                >
                  <div className="flex flex-1 items-center gap-2">
                    <Badge className={`${variantColor} h-3 w-3 p-0`} />
                    <span className="flex-1 truncate">{variant.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {runIdSegment}
                    </span>
                  </div>
                </DropdownMenuCheckboxItem>
              );
            })}
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem
              checked={selectedVariants.length === variants.length}
              onCheckedChange={selectAll}
              className="font-medium"
            >
              Select All
            </DropdownMenuCheckboxItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Display selected variants as badges */}
      <div className="mt-3 flex flex-wrap gap-2">
        {selectedVariants.map((variantId) => {
          const variant = variants.find((v) => v.id === variantId);
          if (!variant) return null;

          const variantColor = getVariantColor(variantId);
          const runIdSegment = getLastUuidSegment(variant.runId);

          return (
            <TooltipProvider key={variantId}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    className={`${variantColor} flex cursor-help items-center gap-1.5 px-2 py-1`}
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
        })}
      </div>
    </div>
  );
}

// Helper function to get the appropriate color class based on variant
export function getVariantColor(variantId: string, isSelected = true) {
  const variant = variants.find((v) => v.id === variantId);
  if (!variant) return "";

  switch (variant.color) {
    case "blue":
      return isSelected
        ? "bg-blue-600 hover:bg-blue-700"
        : "border-blue-600 text-blue-600 hover:bg-blue-50";
    case "purple":
      return isSelected
        ? "bg-purple-600 hover:bg-purple-700"
        : "border-purple-600 text-purple-600 hover:bg-purple-50";
    case "green":
      return isSelected
        ? "bg-green-600 hover:bg-green-700"
        : "border-green-600 text-green-600 hover:bg-green-50";
    default:
      return isSelected
        ? "bg-gray-600 hover:bg-gray-700"
        : "border-gray-600 text-gray-600 hover:bg-gray-50";
  }
}

// Helper function to get color class for the small badge in dropdown
function getVariantColorClass(variantId: string, isSelected = true) {
  const variant = variants.find((v) => v.id === variantId);
  if (!variant) return "";

  switch (variant.color) {
    case "blue":
      return "bg-blue-600";
    case "purple":
      return "bg-purple-600";
    case "green":
      return "bg-green-600";
    default:
      return "bg-gray-600";
  }
}
