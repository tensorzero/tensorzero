import { Suspense, useMemo } from "react";
import { Await, useNavigate, useSearchParams } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import type { MetricsSectionData } from "./metrics-data.server";

interface MetricsSectionProps {
  promise: Promise<MetricsSectionData>;
  locationKey: string;
}

export function MetricsSection({ promise, locationKey }: MetricsSectionProps) {
  return (
    <SectionLayout>
      <Suspense key={`metrics-${locationKey}`} fallback={<MetricsSkeleton />}>
        <Await
          resolve={promise}
          errorElement={
            <>
              <SectionHeader heading="Metrics" />
              <SectionAsyncErrorState defaultMessage="Failed to load metrics data" />
            </>
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
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const metric_name = searchParams.get("metric_name") || "";

  const handleMetricChange = (metric: string) => {
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("metric_name", metric);
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const metricsExcludingDemonstrations = useMemo(
    () => ({
      metrics: metricsWithFeedback.metrics.filter(
        ({ metric_type }) => metric_type !== "demonstration",
      ),
    }),
    [metricsWithFeedback],
  );

  return (
    <>
      <SectionHeader heading="Metrics" />
      <MetricSelector
        metricsWithFeedback={metricsExcludingDemonstrations}
        selectedMetric={metric_name || ""}
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
      <SectionHeader heading="Metrics" />
      <Skeleton className="mb-4 h-10 w-64" />
      <Skeleton className="h-64 w-full" />
    </>
  );
}
