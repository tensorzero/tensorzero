import { Link } from "react-router";
import { AlertDialog } from "~/components/ui/AlertDialog";
import { useConfig } from "~/context/config";
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
  const config = useConfig();
  const functionConfig = config.functions[functionName];
  const variantConfig = functionConfig?.variants[variantName];
  return variantConfig ? (
    <Link
      to={`/observability/functions/${functionName}/variants/${variantName}`}
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
