import { AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface ServerErrorContentProps {
  status?: number;
  message?: string;
  stack?: string;
}

export function ServerErrorContent({
  status,
  message,
  stack,
}: ServerErrorContentProps) {
  return (
    <Card className="max-w-lg border-0 bg-transparent shadow-none">
      <CardHeader className={stack ? "border-b border-neutral-900" : ""}>
        <div className="flex items-center gap-4">
          <AlertTriangle className="h-6 w-6 shrink-0 text-red-400" />
          <div className="min-w-0 flex-1">
            <CardTitle className="font-medium text-neutral-100">
              {status ? `Error ${status}` : "Something Went Wrong"}
            </CardTitle>
            <p className="mt-1.5 text-sm text-neutral-400">
              {message || "An unexpected error occurred."}
            </p>
          </div>
        </div>
      </CardHeader>
      {stack && (
        <CardContent className="flex h-40 flex-col p-6">
          <pre className="min-h-0 flex-1 overflow-auto rounded bg-neutral-900 p-3 font-mono text-xs text-neutral-400">
            {stack}
          </pre>
        </CardContent>
      )}
    </Card>
  );
}
