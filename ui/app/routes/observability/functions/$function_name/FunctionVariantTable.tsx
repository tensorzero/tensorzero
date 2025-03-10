import { Code } from "~/components/ui/code";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import type { VariantCounts } from "~/utils/clickhouse/function";
import { formatDate } from "~/utils/date";
import { VariantLink } from "~/components/function/variant/VariantLink";

type VariantCountsWithMetadata = VariantCounts & {
  type: string;
  weight: number;
};

export default function FunctionVariantTable({
  variant_counts,
  function_name,
}: {
  variant_counts: VariantCountsWithMetadata[];
  function_name: string;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Variant Name</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Weight</TableHead>
          <TableHead>Count</TableHead>
          <TableHead>Last Used</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {variant_counts.length === 0 ? (
          <TableRow className="hover:bg-background-primary">
            <TableCell
              colSpan={5}
              className="px-3 py-8 text-center text-foreground-muted"
            >
              No variants found
            </TableCell>
          </TableRow>
        ) : (
          variant_counts.map((variant_count) => (
            <TableRow
              key={variant_count.variant_name}
              id={variant_count.variant_name}
            >
              <TableCell className="max-w-[200px]">
                <VariantLink
                  variantName={variant_count.variant_name}
                  functionName={function_name}
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {variant_count.variant_name}
                  </code>
                </VariantLink>
              </TableCell>
              <TableCell>
                <Code>{variant_count.type}</Code>
              </TableCell>
              <TableCell>{variant_count.weight}</TableCell>
              <TableCell>{variant_count.count}</TableCell>
              <TableCell>
                {formatDate(new Date(variant_count.last_used))}
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
