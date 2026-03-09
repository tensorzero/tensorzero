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
import { HelpTooltip, docsUrl } from "~/components/ui/HelpTooltip";
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

  // On success the fetcher's Set-Cookie response stores the key in the
  // browser. React Router then auto-revalidates the root loader, which
  // succeeds with the cookie → infraError becomes null → dialog disappears.

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
      <div className="flex flex-col items-center px-6 pt-9 pb-6 text-center">
        <KeyRound className="mb-5 h-8 w-8 text-orange-500 dark:text-orange-400" />
        <h2 className="text-foreground text-lg font-medium">
          TensorZero Gateway requires
          <br />
          an API key
        </h2>
        <ol className="text-muted-foreground mt-4 space-y-2 text-left text-sm">
          <li className="flex items-start gap-2">
            <span className="bg-muted text-muted-foreground flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs">
              1
            </span>
            <span>
              Verify <ErrorInlineCode>TENSORZERO_API_KEY</ErrorInlineCode> is
              set on the UI server
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="bg-muted text-muted-foreground flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs">
              2
            </span>
            <span>Ensure gateway key has not expired or been revoked</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="bg-muted text-muted-foreground flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs">
              3
            </span>
            <span>
              Check gateway logs for authentication details
            </span>
          </li>
        </ol>
      </div>
      <div className="px-6">
        <div className="flex items-center gap-3">
          <div className="border-border flex-1 border-t" />
          <span className="text-muted-foreground/60 text-xs font-medium uppercase">
            or
          </span>
          <div className="border-border flex-1 border-t" />
        </div>
      </div>
      <div className="px-6 pt-4 pb-6">
        <div className="flex items-center gap-1.5">
          <label
            htmlFor="gateway-api-key"
            className="text-foreground text-sm font-medium"
          >
            Authenticate this browser
          </label>
          <HelpTooltip
            link={{
              href: docsUrl("operations/set-up-auth-for-tensorzero"),
              label: "Auth docs",
            }}
          >
            Your API key looks like{" "}
            <code className="text-xs">sk-t0-xxx...-yyy...</code>
          </HelpTooltip>
        </div>
        <div className="mt-2">
          <Input
            id="gateway-api-key"
            type="password"
            placeholder="API key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            disabled={isBusy}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit();
            }}
          />
          {/* Animate height to prevent layout jump when error appears */}
          <div
            className="grid transition-[grid-template-rows] duration-200 ease-out"
            style={{ gridTemplateRows: error ? "1fr" : "0fr" }}
          >
            <p className="overflow-hidden text-destructive text-sm">
              {error && <span className="mt-1.5 block">{error}</span>}
            </p>
          </div>
        </div>
        <Button
          type="button"
          className="mt-4 w-full"
          disabled={isBusy || !apiKey.trim()}
          onClick={handleSubmit}
        >
          {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Connect"}
        </Button>
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
