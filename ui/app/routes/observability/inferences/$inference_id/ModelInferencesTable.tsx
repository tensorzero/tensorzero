import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
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
            <TableRow>
              <TableCell colSpan={2} className="py-4 text-center text-fg-muted">
                No model inferences available.
              </TableCell>
            </TableRow>
          ) : (
            modelInferences.map((inference) => (
              <TableRow
                key={inference.id}
                className="cursor-pointer hover:bg-bg-hover"
                onClick={() => handleRowClick(inference)}
              >
                <TableCell className="max-w-[200px]">
                  <span className="block overflow-hidden text-ellipsis whitespace-nowrap font-mono">
                    {inference.id}
                  </span>
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
        <SheetContent className="w-full overflow-y-auto bg-bg-secondary p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
          {selectedInference && (
            <ModelInferenceItem inference={selectedInference} />
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
