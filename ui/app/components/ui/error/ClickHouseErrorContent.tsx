import { Database } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface ClickHouseErrorContentProps {
  message?: string;
}

export function ClickHouseErrorContent({
  message,
}: ClickHouseErrorContentProps) {
  return (
    <Card className="max-w-lg border-0 bg-transparent shadow-none">
      <CardHeader className="border-b border-neutral-900">
        <div className="flex items-center gap-4">
          <Database className="h-6 w-6 shrink-0 text-red-400" />
          <div className="min-w-0 flex-1">
            <CardTitle className="font-medium text-neutral-100">
              ClickHouse Connection Error
            </CardTitle>
            <p className="mt-1.5 text-sm text-neutral-400">
              {message || "Unable to connect to the ClickHouse database."}
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="h-40 p-6">
        <h4 className="mb-3 text-sm font-medium text-neutral-100">
          What to check:
        </h4>
        <ul className="space-y-2 text-sm text-neutral-400">
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              1
            </span>
            Verify ClickHouse is running and accessible
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              2
            </span>
            <span>
              Check the{" "}
              <code className="rounded bg-neutral-800 px-1 py-0.5 font-mono text-xs">
                CLICKHOUSE_URL
              </code>{" "}
              environment variable
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              3
            </span>
            Review Gateway logs for connection details
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
