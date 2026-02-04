import { Suspense, useMemo } from "react";
import { Await, useNavigate, useSearchParams } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import type { MetricsSectionData } from "./function-data.server";

interface MetricsSectionProps {
  promise: Promise<MetricsSectionData>;
  locationKey: string;
}

export function MetricsSection({ promise, locationKey }: MetricsSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Metrics" />
      <Suspense key={`metrics-${locationKey}`} fallback={<MetricsSkeleton />}>
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load metrics" />
          }
        >
          {(data) => <MetricsContent data={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function MetricsContent({ data }: { data: MetricsSectionData }) {
  const { metricsWithFeedback, variant_performances } = data;
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const metric_name = searchParams.get("metric_name") || "";

  const metricsExcludingDemonstrations = useMemo(
    () => ({
      metrics: metricsWithFeedback.metrics.filter(
        ({ metric_type }) => metric_type !== "demonstration",
      ),
    }),
    [metricsWithFeedback],
  );

  const handleMetricChange = (metric: string) => {
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("metric_name", metric);
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <MetricSelector
        metricsWithFeedback={metricsExcludingDemonstrations}
        selectedMetric={metric_name}
        onMetricChange={handleMetricChange}
      />
      {variant_performances && (
        <VariantPerformance
          variant_performances={variant_performances}
          metric_name={metric_name}
        />
      )}
    </>
  );
}

function MetricsSkeleton() {
  return (
    <>
      <Skeleton className="mb-4 h-10 w-64" />
      <Skeleton className="h-64 w-full" />
    </>
  );
}
