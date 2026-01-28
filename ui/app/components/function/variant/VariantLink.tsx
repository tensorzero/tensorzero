import { Link } from "react-router";
import { useFunctionConfig } from "~/context/config";
import { toVariantUrl } from "~/utils/urls";
import type { ReactNode } from "react";
import { useToast } from "~/hooks/use-toast";

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
  const { toast } = useToast();
  const functionConfig = useFunctionConfig(functionName);
  const variantConfig = functionConfig?.variants[variantName];
  return variantConfig ? (
    <Link to={toVariantUrl(functionName, variantName)}>{children}</Link>
  ) : (
    <button
      type="button"
      onClick={() => {
        toast.error({
          description:
            "This variant is not present in your configuration file.",
        });
      }}
    >
      {children}
    </button>
  );
}
