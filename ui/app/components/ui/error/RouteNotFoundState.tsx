import { Server, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface RouteNotFoundStateProps {
  routeInfo?: string | null;
  onDismiss?: () => void;
}

export function RouteNotFoundState({
  routeInfo,
  onDismiss,
}: RouteNotFoundStateProps) {
  return (
    <Card className="max-w-lg border-0 shadow-none">
      <CardHeader className="border-b">
        <div className="flex items-start gap-4">
          <div className="flex min-w-0 flex-1 items-center gap-4">
            <Server className="h-6 w-6 shrink-0 text-red-600" />
            <div className="min-w-0 flex-1">
              <CardTitle className="font-medium">API Route Not Found</CardTitle>
              <p className="text-muted-foreground mt-1.5 text-sm">
                {routeInfo
                  ? `The gateway returned 404 for: ${routeInfo}`
                  : "The gateway returned 404 for an internal API route."}
              </p>
            </div>
          </div>
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="-mt-4 -mr-4 flex h-8 w-8 shrink-0 items-center justify-center rounded text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              aria-label="Dismiss error"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-6">
        <h4 className="mb-3 text-sm font-medium">What to check:</h4>
        <ul className="text-muted-foreground space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-100 text-xs">
              1
            </span>
            This may indicate a version mismatch between UI and Gateway
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-100 text-xs">
              2
            </span>
            Try refreshing the page or restarting the gateway
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
