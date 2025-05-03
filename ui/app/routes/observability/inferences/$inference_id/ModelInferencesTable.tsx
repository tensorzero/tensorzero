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
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference.server";
import { ModelInferenceItem } from "./ModelInferenceItem";

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
                  <span className="block overflow-hidden font-mono text-ellipsis whitespace-nowrap">
                    {inference.id}
                  </span>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <span className="block overflow-hidden font-mono text-ellipsis whitespace-nowrap">
                    {inference.model_name}
                  </span>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
        <SheetContent className="bg-bg-secondary w-full overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
          {selectedInference && (
            <ModelInferenceItem inference={selectedInference} />
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
