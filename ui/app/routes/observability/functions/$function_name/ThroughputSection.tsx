import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { VariantThroughput } from "~/components/function/variant/VariantThroughput";
import type { ThroughputSectionData } from "./function-data.server";

interface ThroughputSectionProps {
  promise: Promise<ThroughputSectionData>;
  locationKey: string;
}

export function ThroughputSection({
  promise,
  locationKey,
}: ThroughputSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Throughput" />
      <Suspense
        key={`throughput-${locationKey}`}
        fallback={<Skeleton className="h-64 w-full" />}
      >
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load throughput" />
          }
        >
          {(data) => <VariantThroughput variant_throughput={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
