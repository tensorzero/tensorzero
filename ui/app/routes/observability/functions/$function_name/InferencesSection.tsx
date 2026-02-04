import { Suspense } from "react";
import { Await, useAsyncError, useNavigate } from "react-router";
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
import PageButtons from "~/components/utils/PageButtons";
import FunctionInferenceTable from "./FunctionInferenceTable";
import type { InferencesSectionData } from "./function-data.server";

interface InferencesSectionProps {
  promise: Promise<InferencesSectionData>;
  locationKey: string;
}

export function InferencesSection({
  promise,
  locationKey,
}: InferencesSectionProps) {
  return (
    <SectionLayout>
      <Suspense
        key={`inferences-${locationKey}`}
        fallback={<InferencesSkeleton />}
      >
        <Await resolve={promise} errorElement={<InferencesError />}>
          {(data) => <InferencesContent data={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function InferencesContent({ data }: { data: InferencesSectionData }) {
  const { inferences, hasNextPage, hasPreviousPage, count } = data;
  const navigate = useNavigate();

  const topInference = inferences.length > 0 ? inferences[0] : null;
  const bottomInference =
    inferences.length > 0 ? inferences[inferences.length - 1] : null;

  const handleNextPage = () => {
    if (!bottomInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <SectionHeader heading="Inferences" count={count} />
      <FunctionInferenceTable inferences={inferences} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={!hasPreviousPage}
        disableNext={!hasNextPage}
      />
    </>
  );
}

function InferencesTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>ID</TableHead>
        <TableHead>Episode ID</TableHead>
        <TableHead>Variant</TableHead>
        <TableHead>Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

function InferencesSkeleton() {
  return (
    <>
      <SectionHeader heading="Inferences" />
      <Table>
        <InferencesTableHeaders />
        <TableBody>
          {Array.from({ length: 5 }).map((_, i) => (
            <TableRow key={i}>
              <TableCell>
                <Skeleton className="h-4 w-48" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-48" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-24" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-32" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <PageButtons disabled />
    </>
  );
}

function InferencesError() {
  const error = useAsyncError();
  const message = getErrorMessage({
    error,
    fallback: "Failed to load inferences",
  });

  return (
    <>
      <SectionHeader heading="Inferences" />
      <Table>
        <InferencesTableHeaders />
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
      <PageButtons disabled />
    </>
  );
}
