import { Link } from "react-router";
import { Code } from "~/components/ui/code";

type FunctionInfoProps = {
  functionName: string;
  functionType: "chat" | "json";
};

export function FunctionInfo({
  functionName,
  functionType,
}: FunctionInfoProps) {
  return (
    <>
      <dd>
        <Link to={`/observability/functions/${functionName}`}>
          <Code>{functionName}</Code>
        </Link>
      </dd>
      <Code>{functionType}</Code>
    </>
  );
}
