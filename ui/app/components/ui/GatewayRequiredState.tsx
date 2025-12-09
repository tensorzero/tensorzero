import { AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function GatewayRequiredState() {
  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="max-w-md">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <CardTitle>Cannot Connect to TensorZero Gateway</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            The TensorZero UI could not connect to the gateway. Please ensure:
          </p>
          <ul className="text-muted-foreground mt-2 list-disc space-y-1 pl-4 text-sm">
            <li>The TensorZero gateway is running</li>
            <li>
              <code className="bg-muted rounded px-1 py-0.5 font-mono">
                TENSORZERO_GATEWAY_URL
              </code>{" "}
              is correctly configured
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
