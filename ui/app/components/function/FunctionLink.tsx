import { Link } from "react-router";
import { AlertDialog } from "~/components/ui/AlertDialog";
import { useFunctionConfig } from "~/context/config";
import { toFunctionUrl } from "~/utils/urls";
import type { ReactNode } from "react";

type FunctionLinkProps = {
  functionName: string;
  children: ReactNode;
};

export function FunctionLink({ functionName, children }: FunctionLinkProps) {
  const functionConfig = useFunctionConfig(functionName);
  return functionConfig ? (
    <Link to={toFunctionUrl(functionName)}>{children}</Link>
  ) : (
    <AlertDialog
      message="This function is not present in your configuration file."
      trigger={children}
    />
  );
}
