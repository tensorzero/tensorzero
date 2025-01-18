import {
  Table,
  TableBody,
  TableHeader,
  TableHead,
  TableRow,
  TableCell,
} from "~/components/ui/table";
import type { VariantPerformanceRow } from "~/utils/clickhouse/function";
// import { TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "~/components/ui/chart";

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "hsl(var(--chart-1))",
  },
  mobile: {
    label: "Mobile",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig;

export function VariantPerformance({
  variant_performances,
}: {
  variant_performances: VariantPerformanceRow[];
}) {
  return (
    <div className="space-y-8">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Period Start</TableHead>
            <TableHead>Variant Name</TableHead>
            <TableHead>Count</TableHead>
            <TableHead>Average Metric</TableHead>
            <TableHead>Standard Deviation</TableHead>
            <TableHead>95% CI Lower</TableHead>
            <TableHead>95% CI Upper</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {variant_performances.map((performance) => (
            <TableRow
              key={`${performance.period_start}-${performance.variant_name}`}
            >
              <TableCell>{performance.period_start}</TableCell>
              <TableCell>{performance.variant_name}</TableCell>
              <TableCell>{performance.count}</TableCell>
              <TableCell>{performance.avg_metric.toFixed(4)}</TableCell>
              <TableCell>{performance.stdev.toFixed(4)}</TableCell>
              <TableCell>{performance.ci_lower_95.toFixed(4)}</TableCell>
              <TableCell>{performance.ci_upper_95.toFixed(4)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Card>
        <CardHeader>
          <CardTitle>Variant Performance Over Time</CardTitle>
          <CardDescription>
            Showing average metric values by variant
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig}>
            <BarChart data={variant_performances} height={300}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="period_start"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                tickFormatter={(value: string) =>
                  new Date(value).toLocaleDateString()
                }
              />
              <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent indicator="dashed" />}
              />
              <Bar
                dataKey="avg_metric"
                name="Average Metric"
                fill="hsl(var(--primary))"
                radius={4}
              />
            </BarChart>
          </ChartContainer>
        </CardContent>
        <CardFooter className="flex-col items-start gap-2 text-sm">
          <div className="leading-none text-muted-foreground">
            Showing average metric values across variants and time periods
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
