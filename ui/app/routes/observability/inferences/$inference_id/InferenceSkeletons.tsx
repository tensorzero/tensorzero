import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

export function ActionsSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-8" />
    </div>
  );
}

export function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

export function FeedbackTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Metric</TableHead>
        <TableHead>Value</TableHead>
        <TableHead>Tags</TableHead>
        <TableHead>Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

export function FeedbackTableSkeleton() {
  return (
    <Table>
      <FeedbackTableHeaders />
      <TableBody>
        {Array.from({ length: 5 }).map((_, i) => (
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
              <Skeleton className="h-4 w-24" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-28" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function ModelInferencesTableHeaders() {
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

export function ModelInferencesTableSkeleton() {
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
