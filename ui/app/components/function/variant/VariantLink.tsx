import { AlertDialog } from "~/components/ui/AlertDialog";
import { useConfig } from "~/context/config";
import type { ReactNode } from "react";
import { Link } from "~/safe-navigation";

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
      to={[
        "/observability/functions/:function_name/variants/:variant_name",
        {
          function_name: functionName,
          variant_name: variantName,
        },
      ]}
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
