import { AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function GatewayAuthFailedState() {
  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="max-w-md">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <CardTitle>Failed to authenticate with the gateway</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm text-balance">
            The TensorZero UI was able to reach the TensorZero Gateway but
            failed authentication. Please ensure you've set the environment
            variable{" "}
            <code className="bg-muted rounded px-1 py-0.5 font-mono">
              TENSORZERO_API_KEY
            </code>{" "}
            for the TensorZero UI with a valid API key.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
