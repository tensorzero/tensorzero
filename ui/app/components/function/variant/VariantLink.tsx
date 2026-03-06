import { Link } from "react-router";
import { useFunctionConfig } from "~/context/config";
import { useSnapshotHash } from "~/context/snapshot";
import { toVariantUrl } from "~/utils/urls";
import type { ReactNode } from "react";
import { useToast } from "~/hooks/use-toast";
import { DEFAULT_FUNCTION } from "~/utils/constants";

type VariantLinkProps = {
  variantName: string;
  functionName: string;
  children: ReactNode;
  snapshotHash?: string | null;
};

export function VariantLink({
  variantName,
  functionName,
  children,
  snapshotHash: snapshotHashProp,
}: VariantLinkProps) {
  const { toast } = useToast();
  const snapshotHashFromUrl = useSnapshotHash();
  const snapshotHash = snapshotHashProp ?? snapshotHashFromUrl;
  const functionConfig = useFunctionConfig(functionName);
  const variantConfig = functionConfig?.variants[variantName];
  // When viewing a historical snapshot, the variant may not exist in current
  // config — always render a link so navigation stays on the snapshot chain.
  // For DEFAULT_FUNCTION, variants are dynamically generated based on model names
  // and won't be in the config, but the variant detail page handles this.
  const isValidVariant =
    variantConfig || functionName === DEFAULT_FUNCTION || Boolean(snapshotHash);
  return isValidVariant ? (
    <Link to={toVariantUrl(functionName, variantName, snapshotHash)}>
      {children}
    </Link>
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
