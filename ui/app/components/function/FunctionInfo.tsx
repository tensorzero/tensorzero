import { Code } from "~/components/ui/code";
import { useConfig } from "~/context/config";
import { AlertDialog } from "~/components/ui/AlertDialog";
import type { FunctionConfig } from "tensorzero-node";
import { FunctionLink } from "./FunctionLink";

type FunctionInfoProps = {
  functionName: string;
  functionType: FunctionConfig["type"];
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
          <FunctionLink functionName={functionName}>
            <Code>{functionName}</Code>
          </FunctionLink>
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
