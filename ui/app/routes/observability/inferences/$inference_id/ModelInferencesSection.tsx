import { useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { TableErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";

// Shared headers for skeleton and error states
function ModelInferencesTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Model</TableHead>
        <TableHead>Input Tokens</TableHead>
        <TableHead>Output Tokens</TableHead>
        <TableHead>TTFT</TableHead>
        <TableHead>Response Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

// Skeleton
export function ModelInferencesSkeleton() {
  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        {Array.from({ length: 3 }).map((_, i) => (
          <TableRow key={i}>
            <TableCell>
              <Skeleton className="h-4 w-24" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-20" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// Error
export function ModelInferencesSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load model inferences";

  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        <TableRow>
          <TableCell colSpan={6}>
            <TableErrorNotice
              icon={AlertCircle}
              title="Error loading data"
              description={message}
            />
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}
