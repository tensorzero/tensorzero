import { Suspense } from "react";
import { Await, useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import FunctionVariantTable from "./FunctionVariantTable";
import type { VariantsSectionData } from "./variants-data.server";

interface VariantsSectionProps {
  promise: Promise<VariantsSectionData>;
  functionName: string;
  locationKey: string;
}

export function VariantsSection({
  promise,
  functionName,
  locationKey,
}: VariantsSectionProps) {
  return (
    <SectionLayout>
      <Suspense key={`variants-${locationKey}`} fallback={<VariantsSkeleton />}>
        <Await resolve={promise} errorElement={<VariantsError />}>
          {(data) => (
            <VariantsContent data={data} functionName={functionName} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function VariantsContent({
  data,
  functionName,
}: {
  data: VariantsSectionData;
  functionName: string;
}) {
  return (
    <>
      <SectionHeader heading="Variants" />
      <FunctionVariantTable
        variant_counts={data.variant_counts}
        function_name={functionName}
      />
    </>
  );
}

function VariantsTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>Variant Name</TableHead>
        <TableHead>Type</TableHead>
        <TableHead>Count</TableHead>
        <TableHead>Last Used</TableHead>
      </TableRow>
    </TableHeader>
  );
}

function VariantsSkeleton() {
  return (
    <>
      <SectionHeader heading="Variants" />
      <Table>
        <VariantsTableHeaders />
        <TableBody>
          {Array.from({ length: 3 }).map((_, i) => (
            <TableRow key={i}>
              <TableCell>
                <Skeleton className="h-4 w-32" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-24" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-16" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-32" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}

function VariantsError() {
  const error = useAsyncError();
  const message = getErrorMessage({
    error,
    fallback: "Failed to load variants",
  });

  return (
    <>
      <SectionHeader heading="Variants" />
      <Table>
        <VariantsTableHeaders />
        <TableBody>
          <TableRow>
            <TableCell colSpan={4}>
              <TableErrorNotice
                icon={AlertCircle}
                title="Error loading data"
                description={message}
              />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </>
  );
}
