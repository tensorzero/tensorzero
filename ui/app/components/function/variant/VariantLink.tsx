import { Link } from "react-router";
import { AlertDialog } from "~/components/ui/AlertDialog";
import { useFunctionConfig } from "~/context/config";
import type { ReactNode } from "react";

type VariantLinkProps = {
  variantName: string;
  functionName: string;
  children: ReactNode;
};

export function VariantLink({
  variantName,
  functionName,
  children,
}: VariantLinkProps) {
  const functionConfig = useFunctionConfig(functionName);
  const variantConfig = functionConfig?.variants[variantName];
  const encodedVariantName = encodeURIComponent(variantName);
  return variantConfig ? (
    <Link
      to={`/observability/functions/${functionName}/variants/${encodedVariantName}`}
    >
      {children}
    </Link>
  ) : (
    <AlertDialog
      message="This variant is not present in your configuration file."
      trigger={children}
    />
  );
}
