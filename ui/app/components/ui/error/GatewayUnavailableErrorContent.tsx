import { Unplug } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function GatewayUnavailableErrorContent() {
  return (
    <Card className="max-w-lg border-0 bg-transparent shadow-none">
      <CardHeader className="border-b border-neutral-900">
        <div className="flex items-center gap-4">
          <Unplug className="h-6 w-6 shrink-0 text-red-400" />
          <div className="min-w-0 flex-1">
            <CardTitle className="font-medium text-neutral-100">
              Gateway Unavailable
            </CardTitle>
            <p className="mt-1.5 text-sm text-neutral-400">
              Unable to connect to the TensorZero Gateway.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="h-40 p-6">
        <h4 className="mb-3 text-sm font-medium text-neutral-100">
          Troubleshooting steps:
        </h4>
        <ul className="space-y-2 text-sm text-neutral-400">
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              1
            </span>
            Ensure the Gateway is running and accessible
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              2
            </span>
            <span>
              Verify the{" "}
              <code className="rounded bg-neutral-800 px-1 py-0.5 font-mono text-xs">
                TENSORZERO_GATEWAY_URL
              </code>{" "}
              environment variable
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-800 text-xs text-neutral-300">
              3
            </span>
            Check for network connectivity issues
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
