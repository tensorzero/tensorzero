import { Pie, PieChart, Cell } from "recharts";
import { CHART_COLORS } from "~/utils/chart";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
} from "~/components/ui/chart";

export type VariantWeight = {
  variant_name: string;
  weight: number;
};

interface TooltipPayload {
  value: number;
  name: string;
  payload: {
    variant_name: string;
  };
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayload[];
}

function CustomTooltipContent({ active, payload }: TooltipProps) {
  if (!active || !payload || !payload.length) return null;

  const entry = payload[0];

  return (
    <div className="border-border/50 bg-background rounded-lg border px-2.5 py-1.5 text-xs shadow-xl">
      <span className="text-foreground font-mono text-xs">
        {entry.payload.variant_name}
      </span>
    </div>
  );
}

export function ExperimentationPieChart({
  variantWeights,
}: {
  variantWeights: VariantWeight[];
}) {
  // Calculate total weight for percentage calculation
  const totalWeight = variantWeights.reduce(
    (sum, variant) => sum + variant.weight,
    0,
  );

  // Transform data for the pie chart
  const data = variantWeights.map((variant) => ({
    variant_name: variant.variant_name,
    weight: variant.weight,
    percentage: `${((variant.weight / totalWeight) * 100).toFixed(1)}%`,
  }));

  // Create chart config
  const chartConfig: Record<string, { label: string; color: string }> =
    variantWeights.reduce(
      (config, variant, index) => ({
        ...config,
        [variant.variant_name]: {
          label: variant.variant_name,
          color: CHART_COLORS[index % CHART_COLORS.length],
        },
      }),
      {},
    );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Active Variant Weights</CardTitle>
        <CardDescription>
          Distribution of sampling weights across variants
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-80 w-full">
          <PieChart>
            <Pie
              data={data}
              dataKey="weight"
              nameKey="variant_name"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label={({ percentage }) => percentage}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${entry.variant_name}`}
                  fill={CHART_COLORS[index % CHART_COLORS.length]}
                />
              ))}
            </Pie>
            <ChartTooltip content={<CustomTooltipContent />} />
            <ChartLegend
              content={<ChartLegendContent className="font-mono text-xs" />}
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
