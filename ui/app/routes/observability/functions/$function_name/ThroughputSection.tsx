import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { VariantThroughput } from "~/components/function/variant/VariantThroughput";
import type { ThroughputSectionData } from "./throughput-data.server";

interface ThroughputSectionProps {
  throughputData: Promise<ThroughputSectionData>;
  locationKey: string;
}

export function ThroughputSection({
  throughputData,
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
          resolve={throughputData}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load throughput data" />
          }
        >
          {(data) => <VariantThroughput variant_throughput={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
