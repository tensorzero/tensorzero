import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface ParameterCardProps {
  title: string;
  parameters: Record<string, unknown>;
}

export function ParameterCard({ title, parameters }: ParameterCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto rounded-md bg-muted p-4">
          <code className="text-sm">{JSON.stringify(parameters, null, 2)}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
