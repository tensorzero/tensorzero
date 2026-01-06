import { Server } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface RouteNotFoundErrorContentProps {
  routeInfo?: string | null;
}

export function RouteNotFoundErrorContent({
  routeInfo,
}: RouteNotFoundErrorContentProps) {
  return (
    <Card className="max-w-lg border-0 bg-transparent shadow-none">
      <CardHeader className="border-b border-neutral-900">
        <div className="flex items-center gap-4">
          <Server className="h-6 w-6 shrink-0 text-red-400" />
          <div className="min-w-0 flex-1">
            <CardTitle className="font-medium text-neutral-100">
              API Route Not Found
            </CardTitle>
            <p className="mt-1.5 text-sm text-neutral-400">
              {routeInfo
                ? `The Gateway returned 404 for: ${routeInfo}`
                : "The Gateway returned 404 for an internal API route."}
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
            Ensure the UI and Gateway versions are compatible
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              2
            </span>
            Try refreshing the page or restarting the Gateway
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              3
            </span>
            Check Gateway logs for more details
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
