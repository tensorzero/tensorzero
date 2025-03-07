import { Link } from "react-router";
import { Code } from "~/components/ui/code";
import { useConfig } from "~/context/config";
import { AlertDialog } from "~/components/ui/AlertDialog";
import type { VariantType } from "~/utils/config/variant";

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
  const config = useConfig();
  const functionConfig = config.functions[functionName];
  const variantConfig = functionConfig?.variants[variantName];

  return (
    <>
      <dd>
        {variantConfig ? (
          <Link
            to={`/observability/functions/${functionName}/variants/${variantName}`}
          >
            <Code>{variantName}</Code>
          </Link>
        ) : (
          <AlertDialog
            message="This variant is not present in your configuration file"
            trigger={<Code>{variantName}</Code>}
          />
        )}
      </dd>
      {variantType && <Code>{variantType}</Code>}
    </>
  );
}
