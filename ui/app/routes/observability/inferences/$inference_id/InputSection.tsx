import { Suspense } from "react";
import { Await, useAsyncError } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
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
        <Await resolve={promise} errorElement={<InputError />}>
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

// Error
function InputError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load input";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      {message}
    </div>
  );
}
