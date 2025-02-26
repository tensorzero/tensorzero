import { Link } from "react-router";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { FunctionConfig } from "~/utils/config/function";
import type { FunctionCountInfo } from "~/utils/clickhouse/inference";
import { formatDate } from "~/utils/date";
import { Code } from "~/components/ui/code";

export default function FunctionsTable({
  functions,
  countsInfo,
}: {
  functions: Record<string, FunctionConfig>;
  countsInfo: FunctionCountInfo[];
}) {
  // Create a union of all function names from both data sources.
  const functionNamesSet = new Set<string>([
    ...Object.keys(functions),
    ...countsInfo.map((info) => info.function_name),
  ]);

  const mergedFunctions = Array.from(functionNamesSet).map((function_name) => {
    const countInfo = countsInfo.find(
      (info) => info.function_name === function_name,
    );
    const function_config = functions[function_name] || null;

    // Special handling: if the function name is 'tensorzero::default', type is 'chat'
    let type: "chat" | "json" | "?";
    if (function_config) {
      type = function_config.type;
    } else {
      type = "?";
    }

    return {
      function_name,
      count: countInfo ? countInfo.count : 0,
      max_timestamp: countInfo ? countInfo.max_timestamp : "Never",
      type,
    };
  });

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Inference Count</TableHead>
            <TableHead>Last Used</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {mergedFunctions.map(
            ({ function_name, count, max_timestamp, type }) => (
              <TableRow key={function_name} id={function_name}>
                <TableCell className="max-w-[200px] lg:max-w-none">
                  <Link
                    to={`/observability/functions/${function_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {function_name}
                    </code>
                  </Link>
                </TableCell>
                <TableCell>
                  <Code>{type}</Code>
                </TableCell>
                <TableCell>{count}</TableCell>
                <TableCell>
                  {max_timestamp === "Never"
                    ? "Never"
                    : formatDate(new Date(max_timestamp))}
                </TableCell>
              </TableRow>
            ),
          )}
        </TableBody>
      </Table>
    </div>
  );
}
