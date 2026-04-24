import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { VariantUsage } from "~/components/function/variant/VariantUsage";
import type { VariantUsageTimePoint } from "~/types/tensorzero";

interface VariantUsageSectionProps {
  variantUsageData: Promise<VariantUsageTimePoint[]>;
  locationKey: string;
}

export function VariantUsageSection({
  variantUsageData,
}: VariantUsageSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader heading="Variant Usage" />
      <VariantUsage variantUsageDataPromise={variantUsageData} />
    </SectionLayout>
  );
}
