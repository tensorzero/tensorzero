import { Code } from "~/components/ui/code";
import type { VariantConfig } from "~/types/tensorzero";
import { VariantLink } from "./VariantLink";

type VariantInfoProps = {
  variantName: string;
  functionName: string;
  variantType?: VariantConfig["type"];
};

export function VariantInfo({
  variantName,
  functionName,
  variantType,
}: VariantInfoProps) {
  return (
    <>
      <dd>
        <VariantLink variantName={variantName} functionName={functionName}>
          <Code>{variantName}</Code>
        </VariantLink>
      </dd>
      {variantType && <Code>{variantType}</Code>}
    </>
  );
}
