import { AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function AutopilotUnavailableState() {
  return (
    <div className="grid min-h-full place-items-center p-4">
      <Card className="max-w-md">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-amber-500" />
            <CardTitle>TensorZero Autopilot is not configured</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            TensorZero Autopilot features are not available. To enable
            Autopilot, set the following environment variable on the gateway.
          </p>
          <p className="text-muted-foreground mt-4 text-sm text-balance">
            <code className="bg-muted rounded px-1 py-0.5 font-mono">
              TENSORZERO_AUTOPILOT_API_KEY
            </code>
          </p>
          <p className="text-muted-foreground mt-4 text-sm">
            Visit{" "}
            <a
              href="https://www.tensorzero.com"
              target="_blank"
              className="hover:text-primary text-nowrap underline"
            >
              tensorzero.com
            </a>{" "}
            to learn more about TensorZero Autopilot.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
