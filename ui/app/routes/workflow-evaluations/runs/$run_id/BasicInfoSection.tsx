import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import { Await, useAsyncError } from "react-router";
import BasicInfo from "./WorkflowEvaluationRunBasicInfo";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import {
  getErrorMessage,
  SectionErrorNotice,
} from "~/components/ui/error/ErrorContentPrimitives";
import type { BasicInfoData } from "./route.server";

interface BasicInfoSectionProps {
  basicInfoData: Promise<BasicInfoData>;
  locationKey: string;
}

export function BasicInfoSection({
  basicInfoData,
  locationKey,
}: BasicInfoSectionProps) {
  return (
    <Suspense
      key={`basic-info-${locationKey}`}
      fallback={<BasicInfoSkeleton />}
    >
      <Await resolve={basicInfoData} errorElement={<BasicInfoError />}>
        {(data) => (
          <BasicInfo
            workflowEvaluationRun={data.workflowEvaluationRun}
            count={data.count}
          />
        )}
      </Await>
    </Suspense>
  );
}

function BasicInfoSkeleton() {
  return <BasicInfoLayoutSkeleton rows={7} />;
}

function BasicInfoError() {
  const error = useAsyncError();
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading run info"
      description={getErrorMessage({
        error,
        fallback: "Failed to load workflow evaluation run info",
      })}
    />
  );
}
