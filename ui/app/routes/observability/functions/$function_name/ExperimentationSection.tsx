import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { FunctionExperimentation } from "./FunctionExperimentation";
import type { FunctionConfig } from "~/types/tensorzero";
import type { ExperimentationSectionData } from "./experimentation-data.server";

interface ExperimentationSectionProps {
  promise: Promise<ExperimentationSectionData>;
  functionConfig: FunctionConfig;
  functionName: string;
  locationKey: string;
}

export function ExperimentationSection({
  promise,
  functionConfig,
  functionName,
  locationKey,
}: ExperimentationSectionProps) {
  return (
    <Suspense
      key={`experimentation-${locationKey}`}
      fallback={
        <SectionLayout>
          <SectionHeader heading="Experimentation" />
          <Skeleton className="h-32 w-full" />
        </SectionLayout>
      }
    >
      <Await
        resolve={promise}
        errorElement={
          <SectionLayout>
            <SectionHeader heading="Experimentation" />
            <SectionAsyncErrorState defaultMessage="Failed to load experimentation data" />
          </SectionLayout>
        }
      >
        {(data) => (
          <SectionLayout>
            <SectionHeader heading="Experimentation" />
            <FunctionExperimentation
              functionConfig={functionConfig}
              functionName={functionName}
              feedbackTimeseries={data.feedback_timeseries}
              variantSamplingProbabilities={
                data.variant_sampling_probabilities
              }
            />
          </SectionLayout>
        )}
      </Await>
    </Suspense>
  );
}
