import { type VariantConfig } from "~/utils/config/variant";
import { DataTable } from "~/components/ui/data-table";
import {
  createVariantColumns,
  type VariantTableRow,
} from "./variant-table-columns";
import { type VisibilityState } from "@tanstack/react-table";

// Essential columns that should always be visible
const ESSENTIAL_COLUMNS = ["name", "type", "weight"];

// Analyze variant data to determine smart default column visibility
function computeDefaultColumnVisibility(
  variants: [string, VariantConfig][]
): VisibilityState {
  if (variants.length === 0) {
    return {}; // Show all columns if no data
  }

  const columnVisibility: VisibilityState = {};
  const columns = createVariantColumns();

  columns.forEach((column) => {
    const columnId = column.id as string;
    
    // Always show essential columns
    if (ESSENTIAL_COLUMNS.includes(columnId)) {
      columnVisibility[columnId] = true;
      return;
    }

    // Analyze if this column has meaningful variation across variants
    const values = variants.map(([_, config]) => {
      switch (columnId) {
        case "model":
          return "model" in config ? config.model : undefined;
        case "temperature":
          return "temperature" in config ? config.temperature : undefined;
        case "max_tokens":
          return "max_tokens" in config ? config.max_tokens : undefined;
        case "timeout_s":
          return "timeout_s" in config ? config.timeout_s : undefined;
        case "candidates":
          return "candidates" in config ? config.candidates?.length || 0 : undefined;
        case "embedding_model":
          return "embedding_model" in config ? config.embedding_model : undefined;
        case "k":
          return "k" in config ? config.k : undefined;
        case "json_mode":
          return "json_mode" in config ? config.json_mode : undefined;
        default:
          return undefined;
      }
    });

    // Filter out undefined values
    const definedValues = values.filter(v => v !== undefined);
    
    // Hide column if:
    // 1. No variants have this property, OR
    // 2. All variants have the same value for this property
    const shouldHide = definedValues.length === 0 || 
                      (definedValues.length > 1 && new Set(definedValues).size === 1);
    
    columnVisibility[columnId] = !shouldHide;
  });

  return columnVisibility;
}

export const VariantTable: React.FC<{
  variants: [string, VariantConfig][];
  onVariantSelect?: (variantName: string | null) => void;
  selectedVariant?: string;
}> = ({ variants, onVariantSelect, selectedVariant }) => {
  // Transform variants data for the table
  const tableData: VariantTableRow[] = variants.map(([name, config]) => ({
    id: name,
    name,
    config,
  }));

  const columns = createVariantColumns();
  
  // Compute smart default column visibility
  const initialColumnVisibility = computeDefaultColumnVisibility(variants);

  const handleRowSelect = (selectedRow: VariantTableRow | null) => {
    if (onVariantSelect) {
      onVariantSelect(selectedRow?.name || null);
    }
  };

  return (
    <DataTable
      columns={columns}
      data={tableData}
      onRowSelect={handleRowSelect}
      selectedRowId={selectedVariant}
      initialColumnVisibility={initialColumnVisibility}
    />
  );
};
