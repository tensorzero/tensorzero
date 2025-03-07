import { Link } from "react-router";
import { Code } from "~/components/ui/code";
import { useConfig } from "~/context/config";
import { AlertDialog } from "~/components/ui/AlertDialog";

type FunctionInfoProps = {
  functionName: string;
  functionType: "chat" | "json";
};

export function FunctionInfo({
  functionName,
  functionType,
}: FunctionInfoProps) {
  const config = useConfig();
  const functionConfig = config.functions[functionName];

  return (
    <>
      <dd>
        {functionConfig ? (
          <Link to={`/observability/functions/${functionName}`}>
            <Code>{functionName}</Code>
          </Link>
        ) : (
          <AlertDialog
            message="This function is not present in your configuration file"
            trigger={<Code>{functionName}</Code>}
          />
        )}
      </dd>
      <Code>{functionType}</Code>
    </>
  );
}
