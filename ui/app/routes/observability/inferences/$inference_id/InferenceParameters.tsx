import { Card, CardContent } from "~/components/ui/card";

interface ParameterCardProps {
  parameters: Record<string, unknown>;
}

export function ParameterCard({ parameters }: ParameterCardProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <pre className="overflow-x-auto rounded-md bg-muted p-4">
          <code className="text-sm">{JSON.stringify(parameters, null, 2)}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
