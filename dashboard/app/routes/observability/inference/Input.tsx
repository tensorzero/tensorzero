import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import type { Input } from "~/utils/clickhouse/common";

interface InputProps {
  input: Input;
}

export default function Input({ input }: InputProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Input</CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto rounded-md bg-muted p-4">
          <code className="text-sm">{JSON.stringify(input, null, 2)}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
