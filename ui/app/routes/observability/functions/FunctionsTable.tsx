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
          {countsInfo.map((countInfo) => {
            const function_config = functions[countInfo.function_name];
            if (!function_config) {
              console.warn(
                `No function config found for ${countInfo.function_name}`,
              );
              return null;
            }
            return (
              <TableRow
                key={countInfo.function_name}
                id={countInfo.function_name}
              >
                <TableCell className="max-w-[200px] lg:max-w-none">
                  <a
                    href={`/observability/function/${countInfo.function_name}`}
                    className="block no-underline"
                  >
                    <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                      {countInfo.function_name}
                    </code>
                  </a>
                </TableCell>
                <TableCell>
                  <Code>{function_config.type}</Code>
                </TableCell>
                <TableCell>{countInfo.count}</TableCell>
                <TableCell>
                  {formatDate(new Date(countInfo.max_timestamp))}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
