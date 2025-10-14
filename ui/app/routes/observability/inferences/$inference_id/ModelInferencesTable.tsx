import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { ModelInferenceItem } from "./ModelInferenceItem";
import { TableItemShortUuid } from "~/components/ui/TableItems";

interface ModelInferencesTableProps {
  modelInferences: ParsedModelInferenceRow[];
}

export function ModelInferencesTable({
  modelInferences,
}: ModelInferencesTableProps) {
  const [selectedInference, setSelectedInference] =
    useState<ParsedModelInferenceRow | null>(null);
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const handleRowClick = (inference: ParsedModelInferenceRow) => {
    setSelectedInference(inference);
    setIsSheetOpen(true);
  };

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Model</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {modelInferences.length === 0 ? (
            <TableEmptyState message="No model inferences available" />
          ) : (
            modelInferences.map((inference) => (
              <TableRow
                key={inference.id}
                className="hover:bg-bg-hover cursor-pointer"
                onClick={() => handleRowClick(inference)}
              >
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid id={inference.id} />
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <span className="block overflow-hidden text-ellipsis whitespace-nowrap font-mono">
                    {inference.model_name}
                  </span>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
        <SheetContent className="bg-bg-secondary overflow-y-auto p-0">
          {selectedInference && (
            <ModelInferenceItem inference={selectedInference} />
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
