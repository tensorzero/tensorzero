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
  InlineCode,
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
    <ErrorContentCard>
      <ErrorContentHeader
        icon={Unplug}
        title="Gateway Unavailable"
        description="Unable to connect to the TensorZero Gateway."
      />
      <TroubleshootingSection heading="Troubleshooting steps:">
        <>Ensure the Gateway is running and accessible</>
        <>
          Verify the <InlineCode>TENSORZERO_GATEWAY_URL</InlineCode> environment
          variable
        </>
        <>Check for network connectivity issues</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function GatewayAuthContent() {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={KeyRound}
        title="Authentication Failed"
        description="Unable to authenticate with the TensorZero Gateway."
      />
      <TroubleshootingSection>
        <>
          Verify <InlineCode>TENSORZERO_API_KEY</InlineCode> is set correctly
        </>
        <>Ensure the API key has not expired or been revoked</>
        <>Check Gateway logs for authentication details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function RouteNotFoundContent({ routeInfo }: { routeInfo?: string }) {
  return (
    <ErrorContentCard>
      <ErrorContentHeader
        icon={Server}
        title="API Route Not Found"
        description={
          routeInfo
            ? `The Gateway returned 404 for: ${routeInfo}`
            : "The Gateway returned 404 for an internal API route."
        }
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
          Check the <InlineCode>CLICKHOUSE_URL</InlineCode> environment variable
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
        title={status ? `Error ${status}` : "Something Went Wrong"}
        description={message || "An unexpected error occurred."}
        showBorder={Boolean(stack)}
      />
      {stack && <StackTraceContent stack={stack} />}
    </ErrorContentCard>
  );
}
