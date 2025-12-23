import { Code } from "~/components/ui/code";
import { useFunctionConfig } from "~/context/config";
import { AlertDialog } from "~/components/ui/AlertDialog";
import type { FunctionConfig } from "~/types/tensorzero";
import { FunctionLink } from "./FunctionLink";

type FunctionInfoProps = {
  functionName: string;
  functionType: FunctionConfig["type"];
};

export function FunctionInfo({
  functionName,
  functionType,
}: FunctionInfoProps) {
  const functionConfig = useFunctionConfig(functionName);

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
