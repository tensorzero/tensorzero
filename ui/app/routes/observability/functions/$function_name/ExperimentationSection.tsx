import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { FunctionExperimentation } from "./FunctionExperimentation";
import type { FunctionConfig } from "~/types/tensorzero";
import type { ExperimentationSectionData } from "./function-data.server";

interface ExperimentationSectionProps {
  promise: Promise<ExperimentationSectionData>;
  functionName: string;
  functionConfig: FunctionConfig;
  locationKey: string;
}

export function ExperimentationSection({
  promise,
  functionName,
  functionConfig,
  locationKey,
}: ExperimentationSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Experimentation" />
      <Suspense
        key={`experimentation-${locationKey}`}
        fallback={<Skeleton className="h-32 w-full" />}
      >
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load experimentation data" />
          }
        >
          {(data) => (
            <FunctionExperimentation
              functionConfig={functionConfig}
              functionName={functionName}
              feedbackTimeseries={data.feedback_timeseries}
              variantSamplingProbabilities={data.variant_sampling_probabilities}
            />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
