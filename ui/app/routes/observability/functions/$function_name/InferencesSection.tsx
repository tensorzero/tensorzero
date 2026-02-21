import { Suspense } from "react";
import { Await, useNavigate } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import PageButtons from "~/components/utils/PageButtons";
import FunctionInferenceTable from "./FunctionInferenceTable";
import type { InferencesSectionData } from "./inferences-data.server";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

interface InferencesSectionProps {
  inferencesData: Promise<InferencesSectionData>;
  countPromise: Promise<number>;
  locationKey: string;
}

export function InferencesSection({
  inferencesData,
  countPromise,
  locationKey,
}: InferencesSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Inferences" count={countPromise} />
      <Suspense
        key={`inferences-${locationKey}`}
        fallback={<InferencesSkeleton />}
      >
        <Await resolve={inferencesData} errorElement={<InferencesError />}>
          {(data) => <InferencesContent data={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function InferencesContent({ data }: { data: InferencesSectionData }) {
  const { inferences, hasNextInferencePage, hasPreviousInferencePage } = data;

  const navigate = useNavigate();

  const topInference = inferences.length > 0 ? inferences[0] : null;
  const bottomInference =
    inferences.length > 0 ? inferences[inferences.length - 1] : null;

  const handleNextInferencePage = () => {
    if (!bottomInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <FunctionInferenceTable inferences={inferences} />
      <PageButtons
        onPreviousPage={handlePreviousInferencePage}
        onNextPage={handleNextInferencePage}
        disablePrevious={!hasPreviousInferencePage}
        disableNext={!hasNextInferencePage}
      />
    </>
  );
}

function InferencesSkeleton() {
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {[1, 2, 3, 4, 5].map((i) => (
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
  return (
    <>
      <SectionAsyncErrorState defaultMessage="Failed to load inferences" />
      <PageButtons disabled />
    </>
  );
}
