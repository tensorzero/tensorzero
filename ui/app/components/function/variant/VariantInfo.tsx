import { Code } from "~/components/ui/code";
import type { VariantType } from "~/utils/config/variant";
import { VariantLink } from "./VariantLink";

type VariantInfoProps = {
  variantName: string;
  functionName: string;
  variantType?: VariantType;
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
