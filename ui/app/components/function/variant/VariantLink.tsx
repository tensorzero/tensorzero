import { Link } from "react-router";
import { useFunctionConfig } from "~/context/config";
import { toVariantUrl } from "~/utils/urls";
import type { ReactNode } from "react";
import { useToast } from "~/hooks/use-toast";
import { DEFAULT_FUNCTION } from "~/utils/constants";

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
  // For DEFAULT_FUNCTION, variants are dynamically generated based on model names
  // and won't be in the config, but the variant detail page handles this
  const isValidVariant = variantConfig || functionName === DEFAULT_FUNCTION;
  return isValidVariant ? (
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
