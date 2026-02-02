import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { InputElement } from "~/components/input_output/InputElement";
import type { Input } from "~/types/tensorzero";

// Section - self-contained with Suspense/Await
interface InputSectionProps {
  promise: Promise<Input>;
  locationKey: string;
}

export function InputSection({ promise, locationKey }: InputSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Input" />
      <Suspense key={`input-${locationKey}`} fallback={<InputSkeleton />}>
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load input" />
          }
        >
          {(input) => <InputElement input={input} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

// Skeleton
function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}
