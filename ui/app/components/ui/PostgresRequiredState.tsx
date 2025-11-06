import { AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function PostgresRequiredState() {
  return (
    <div className="flex items-center justify-center p-4">
      <Card className="max-w-md">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-yellow-500" />
            <CardTitle>Postgres Credentials Missing</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            A Postgres database connection is required to access this feature.
            Please set the{" "}
            <code className="bg-muted rounded px-1 py-0.5 font-mono">
              TENSORZERO_POSTGRES_URL
            </code>{" "}
            environment variable.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
