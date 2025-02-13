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
  // Merge the functions and countsInfo data. For functions not present in countsInfo,
  // we default the count to 0 and max_timestamp to "unused".
  // NOTE: We do not include functions that are not in the functions config.
  const mergedFunctions = Object.keys(functions).map((function_name) => {
    const countInfo = countsInfo.find(
      (info) => info.function_name === function_name,
    );
    return {
      function_name,
      count: countInfo ? countInfo.count : 0,
      max_timestamp: countInfo ? countInfo.max_timestamp : "Never",
      function_config: functions[function_name],
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
            ({ function_name, count, max_timestamp, function_config }) => (
              <TableRow key={function_name} id={function_name}>
                <TableCell className="max-w-[200px] lg:max-w-none">
                  <a
                    href={`/observability/function/${function_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {function_name}
                    </code>
                  </a>
                </TableCell>
                <TableCell>
                  <Code>{function_config.type}</Code>
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
