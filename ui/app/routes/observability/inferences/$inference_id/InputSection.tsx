import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { InputElement } from "~/components/input_output/InputElement";
import type { Input } from "~/types/tensorzero";

interface InputSectionProps {
  promise: Promise<Input | undefined>;
  locationKey: string;
}

export function InputSection({ promise, locationKey }: InputSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Input" />
      <Suspense
        key={`input-${locationKey}`}
        fallback={<Skeleton className="h-32 w-full" />}
      >
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load input" />
          }
        >
          {(input) => input && <InputElement input={input} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
