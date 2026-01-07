import {
  AlertTriangle,
  Database,
  KeyRound,
  Server,
  Unplug,
} from "lucide-react";
import {
  BoundaryErrorType,
  type ClassifiedError,
} from "~/utils/tensorzero/errors";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorContext,
  ErrorInlineCode,
  StackTraceContent,
  TroubleshootingSection,
} from "./ErrorContentPrimitives";

interface ErrorContentProps {
  error: ClassifiedError;
}

export function ErrorContent({ error }: ErrorContentProps) {
  switch (error.type) {
    case BoundaryErrorType.GatewayUnavailable:
      return <GatewayUnavailableContent />;
    case BoundaryErrorType.GatewayAuthFailed:
      return <GatewayAuthContent />;
    case BoundaryErrorType.RouteNotFound:
      return <RouteNotFoundContent routeInfo={error.routeInfo} />;
    case BoundaryErrorType.ClickHouseConnection:
      return <ClickHouseContent message={error.message} />;
    case BoundaryErrorType.ServerError:
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
    <ErrorContentCard context={ErrorContext.App}>
      <ErrorContentHeader
        icon={Unplug}
        title="Gateway Unavailable"
        description="Unable to connect to the TensorZero Gateway."
        context={ErrorContext.App}
      />
      <TroubleshootingSection context={ErrorContext.App}>
        <>Ensure the Gateway is running and accessible</>
        <>
          Verify the{" "}
          <ErrorInlineCode context={ErrorContext.App}>
            TENSORZERO_GATEWAY_URL
          </ErrorInlineCode>{" "}
          environment variable
        </>
        <>Check for network connectivity issues</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function GatewayAuthContent() {
  return (
    <ErrorContentCard context={ErrorContext.App}>
      <ErrorContentHeader
        icon={KeyRound}
        title="Authentication Failed"
        description="Unable to authenticate with the TensorZero Gateway."
        context={ErrorContext.App}
      />
      <TroubleshootingSection context={ErrorContext.App}>
        <>
          Verify{" "}
          <ErrorInlineCode context={ErrorContext.App}>
            TENSORZERO_API_KEY
          </ErrorInlineCode>{" "}
          is set correctly
        </>
        <>Ensure the API key has not expired or been revoked</>
        <>Check Gateway logs for authentication details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function RouteNotFoundContent({ routeInfo }: { routeInfo: string }) {
  return (
    <ErrorContentCard context={ErrorContext.App}>
      <ErrorContentHeader
        icon={Server}
        title="API Route Not Found"
        description={`The Gateway returned 404 for: ${routeInfo}`}
        context={ErrorContext.App}
      />
      <TroubleshootingSection context={ErrorContext.App}>
        <>Ensure the UI and Gateway versions are compatible</>
        <>Try refreshing the page or restarting the Gateway</>
        <>Check Gateway logs for more details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function ClickHouseContent({ message }: { message?: string }) {
  return (
    <ErrorContentCard context={ErrorContext.App}>
      <ErrorContentHeader
        icon={Database}
        title="ClickHouse Connection Error"
        description={message || "Unable to connect to the ClickHouse database."}
        context={ErrorContext.App}
      />
      <TroubleshootingSection context={ErrorContext.App}>
        <>Verify ClickHouse is running and accessible</>
        <>
          Check the{" "}
          <ErrorInlineCode context={ErrorContext.App}>
            CLICKHOUSE_URL
          </ErrorInlineCode>{" "}
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
    <ErrorContentCard context={ErrorContext.App}>
      <ErrorContentHeader
        icon={AlertTriangle}
        title={status ? `Error ${status}` : "Something Went Wrong"}
        description={message || "An unexpected error occurred."}
        context={ErrorContext.App}
      />
      {stack && <StackTraceContent stack={stack} context={ErrorContext.App} />}
    </ErrorContentCard>
  );
}
