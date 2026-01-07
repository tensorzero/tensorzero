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
  ErrorInlineCode,
  ErrorStyle,
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
    <ErrorContentCard variant={ErrorStyle.Dark}>
      <ErrorContentHeader
        icon={Unplug}
        title="Gateway Unavailable"
        description="Unable to connect to the TensorZero Gateway."
        variant={ErrorStyle.Dark}
      />
      <TroubleshootingSection variant={ErrorStyle.Dark}>
        <>Ensure the Gateway is running and accessible</>
        <>
          Verify the{" "}
          <ErrorInlineCode variant={ErrorStyle.Dark}>
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
    <ErrorContentCard variant={ErrorStyle.Dark}>
      <ErrorContentHeader
        icon={KeyRound}
        title="Authentication Failed"
        description="Unable to authenticate with the TensorZero Gateway."
        variant={ErrorStyle.Dark}
      />
      <TroubleshootingSection variant={ErrorStyle.Dark}>
        <>
          Verify{" "}
          <ErrorInlineCode variant={ErrorStyle.Dark}>
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
    <ErrorContentCard variant={ErrorStyle.Dark}>
      <ErrorContentHeader
        icon={Server}
        title="API Route Not Found"
        description={`The Gateway returned 404 for: ${routeInfo}`}
        variant={ErrorStyle.Dark}
      />
      <TroubleshootingSection variant={ErrorStyle.Dark}>
        <>Ensure the UI and Gateway versions are compatible</>
        <>Try refreshing the page or restarting the Gateway</>
        <>Check Gateway logs for more details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function ClickHouseContent({ message }: { message?: string }) {
  return (
    <ErrorContentCard variant={ErrorStyle.Dark}>
      <ErrorContentHeader
        icon={Database}
        title="ClickHouse Connection Error"
        description={message || "Unable to connect to the ClickHouse database."}
        variant={ErrorStyle.Dark}
      />
      <TroubleshootingSection variant={ErrorStyle.Dark}>
        <>Verify ClickHouse is running and accessible</>
        <>
          Check the{" "}
          <ErrorInlineCode variant={ErrorStyle.Dark}>
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
    <ErrorContentCard variant={ErrorStyle.Dark}>
      <ErrorContentHeader
        icon={AlertTriangle}
        title={status ? `Error ${status}` : "Something Went Wrong"}
        description={message || "An unexpected error occurred."}
        showBorder={Boolean(stack)}
        variant={ErrorStyle.Dark}
      />
      {stack && <StackTraceContent stack={stack} variant={ErrorStyle.Dark} />}
    </ErrorContentCard>
  );
}
