import { AlertDialog } from "~/components/ui/AlertDialog";
import { useConfig } from "~/context/config";
import type { ReactNode } from "react";
import { Link } from "~/safe-navigation";

type FunctionLinkProps = {
  functionName: string;
  children: ReactNode;
};

export function FunctionLink({ functionName, children }: FunctionLinkProps) {
  const config = useConfig();
  const functionConfig = config.functions[functionName];
  return functionConfig ? (
    <Link
      to={[
        `/observability/functions/:function_name`,
        { function_name: functionName },
      ]}
    >
      {children}
    </Link>
  ) : (
    <AlertDialog
      message="This function is not present in your configuration file."
      trigger={children}
    />
  );
}
