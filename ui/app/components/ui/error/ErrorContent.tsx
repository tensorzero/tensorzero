import { useState } from "react";
import {
  AlertTriangle,
  Database,
  FileQuestion,
  KeyRound,
  Loader2,
  Server,
  Unplug,
} from "lucide-react";
import { isRouteErrorResponse, useFetcher } from "react-router";
import {
  InfraErrorType,
  type ClassifiedError,
  getPageErrorInfo,
} from "~/utils/tensorzero/errors";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorInlineCode,
  PageErrorNotice,
  StackTraceContent,
  TroubleshootingSection,
} from "./ErrorContentPrimitives";

interface ErrorContentProps {
  error: ClassifiedError;
}

export function ErrorContent({ error }: ErrorContentProps) {
  switch (error.type) {
    case InfraErrorType.GatewayUnavailable:
      return <GatewayUnavailableContent />;
    case InfraErrorType.GatewayAuthFailed:
      return <GatewayAuthContent />;
    case InfraErrorType.GatewayEndpointNotFound:
      return <GatewayEndpointNotFoundContent routeInfo={error.routeInfo} />;
    case InfraErrorType.ClickHouseUnavailable:
      return <ClickHouseContent message={error.message} />;
    case InfraErrorType.ServerError:
      return (
        <ServerErrorContent
          status={error.status}
          message={error.message}
          stack={error.stack}
        />
      );
    default: {
      const _exhaustiveCheck: never = error;
      return <ServerErrorContent />;
    }
  }
}

function GatewayUnavailableContent() {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={Unplug}
        title="Gateway Unavailable"
        description="Unable to connect to the TensorZero Gateway."
      />
      <TroubleshootingSection>
        <>Ensure the Gateway is running and accessible</>
        <>
          Verify the <ErrorInlineCode>TENSORZERO_GATEWAY_URL</ErrorInlineCode>{" "}
          environment variable
        </>
        <>Check for network connectivity issues</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function GatewayAuthContent() {
  const fetcher = useFetcher<{ success?: boolean; error?: string }>();
  const [apiKey, setApiKey] = useState("");
  const isBusy = fetcher.state !== "idle";
  const error = fetcher.data?.error;

  const handleSubmit = () => {
    if (apiKey.trim()) {
      fetcher.submit(
        { apiKey },
        { method: "post", action: "/api/auth/set_gateway_key" },
      );
    }
  };

  return (
    <ErrorContentCard className="w-[22rem]">
      <div className="flex flex-col items-center px-6 pt-9 pb-8 text-center">
        <KeyRound className="mb-5 h-8 w-8 text-orange-500 dark:text-orange-400" />
        <h2 className="text-foreground text-lg font-medium">
          Connect to TensorZero
        </h2>
        <p className="text-muted-foreground mt-1.5 text-sm">
          Enter your TensorZero API key or set it as an environment variable on
          the UI server.
        </p>
      </div>
      <div className="space-y-4 px-6 pb-6">
        <div>
          <label
            htmlFor="gateway-api-key"
            className="text-foreground mb-1.5 block text-sm font-medium"
          >
            API Key
          </label>
          <Input
            id="gateway-api-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            disabled={isBusy}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit();
            }}
          />
        </div>
        <Button
          type="button"
          className="w-full"
          disabled={isBusy || !apiKey.trim()}
          onClick={handleSubmit}
        >
          {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Connect"}
        </Button>
        {error && <p className="text-destructive text-sm">{error}</p>}
      </div>
    </ErrorContentCard>
  );
}

function GatewayEndpointNotFoundContent({ routeInfo }: { routeInfo: string }) {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={Server}
        title="Gateway Endpoint Not Found"
        description={`The Gateway returned 404 for: ${routeInfo}`}
      />
      <TroubleshootingSection>
        <>Ensure the UI and Gateway versions are compatible</>
        <>Try refreshing the page or restarting the Gateway</>
        <>Check Gateway logs for more details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function ClickHouseContent({ message }: { message?: string }) {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={Database}
        title="ClickHouse Connection Error"
        description={message || "Unable to connect to the ClickHouse database."}
      />
      <TroubleshootingSection>
        <>Verify ClickHouse is running and accessible</>
        <>
          Check the <ErrorInlineCode>TENSORZERO_CLICKHOUSE_URL</ErrorInlineCode>{" "}
          environment variable
        </>
        <>Review Gateway logs for connection details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function ServerErrorContent({
  status,
  message,
  stack,
}: {
  status?: number;
  message?: string;
  stack?: string;
}) {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={AlertTriangle}
        title={status ? `Error ${status}` : "Something went wrong"}
        description={message || "An unexpected error occurred."}
      />
      {stack && <StackTraceContent stack={stack} />}
    </ErrorContentCard>
  );
}

export function PageNotFound() {
  return (
    <PageErrorNotice
      icon={FileQuestion}
      title="Page Not Found"
      description="The page you're looking for doesn't exist."
      muted
    />
  );
}

interface PageErrorContentProps {
  error: unknown;
}

/**
 * Renders page-level errors inline.
 * For layout ErrorBoundaries, use LayoutErrorBoundary instead - it handles
 * both infra errors (dialog) and page errors (inline) automatically.
 */
export function PageErrorContent({ error }: PageErrorContentProps) {
  // Client 404s (no route matched) have no data - show generic "Page Not Found"
  // Resource 404s (inference/dataset/etc not found) have data - show specific error
  if (isRouteErrorResponse(error) && error.status === 404 && !error.data) {
    return <PageNotFound />;
  }

  const { title, message, status } = getPageErrorInfo(error);

  return (
    <PageErrorNotice
      icon={AlertTriangle}
      title={status ? `Error ${status}` : title}
      description={message}
    />
  );
}
