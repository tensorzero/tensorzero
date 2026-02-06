import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { FunctionExperimentation } from "./FunctionExperimentation";
import type { FunctionConfig } from "~/types/tensorzero";
import type { ExperimentationSectionData } from "./experimentation-data.server";

interface ExperimentationSectionProps {
  promise: Promise<ExperimentationSectionData | undefined>;
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
    <SectionLayout>
      <Suspense
        key={`experimentation-${locationKey}`}
        fallback={
          <>
            <SectionHeader heading="Experimentation" />
            <Skeleton className="h-32 w-full" />
          </>
        }
      >
        <Await
          resolve={promise}
          errorElement={
            <>
              <SectionHeader heading="Experimentation" />
              <SectionAsyncErrorState defaultMessage="Failed to load experimentation data" />
            </>
          }
        >
          {(data) =>
            data ? (
              <>
                <SectionHeader heading="Experimentation" />
                <FunctionExperimentation
                  functionConfig={functionConfig}
                  functionName={functionName}
                  feedbackTimeseries={data.feedback_timeseries}
                  variantSamplingProbabilities={
                    data.variant_sampling_probabilities
                  }
                />
              </>
            ) : null
          }
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
