import {
  AlertTriangle,
  Database,
  KeyRound,
  Server,
  Unplug,
} from "lucide-react";
import {
  InfraErrorType,
  type ClassifiedError,
} from "~/utils/tensorzero/errors";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorScope,
  ErrorInlineCode,
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
    case InfraErrorType.GatewayRouteNotFound:
      return <GatewayRouteNotFoundContent routeInfo={error.routeInfo} />;
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
    <ErrorContentCard scope={ErrorScope.App}>
      <ErrorContentHeader
        icon={Unplug}
        title="Gateway Unavailable"
        description="Unable to connect to the TensorZero Gateway."
        scope={ErrorScope.App}
      />
      <TroubleshootingSection scope={ErrorScope.App}>
        <>Ensure the Gateway is running and accessible</>
        <>
          Verify the{" "}
          <ErrorInlineCode scope={ErrorScope.App}>
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
    <ErrorContentCard scope={ErrorScope.App}>
      <ErrorContentHeader
        icon={KeyRound}
        title="Authentication Failed"
        description="Unable to authenticate with the TensorZero Gateway."
        scope={ErrorScope.App}
      />
      <TroubleshootingSection scope={ErrorScope.App}>
        <>
          Verify{" "}
          <ErrorInlineCode scope={ErrorScope.App}>
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

function GatewayRouteNotFoundContent({ routeInfo }: { routeInfo: string }) {
  return (
    <ErrorContentCard scope={ErrorScope.App}>
      <ErrorContentHeader
        icon={Server}
        title="Gateway Route Not Found"
        description={`The Gateway returned 404 for: ${routeInfo}`}
        scope={ErrorScope.App}
      />
      <TroubleshootingSection scope={ErrorScope.App}>
        <>Ensure the UI and Gateway versions are compatible</>
        <>Try refreshing the page or restarting the Gateway</>
        <>Check Gateway logs for more details</>
      </TroubleshootingSection>
    </ErrorContentCard>
  );
}

function ClickHouseContent({ message }: { message?: string }) {
  return (
    <ErrorContentCard scope={ErrorScope.App}>
      <ErrorContentHeader
        icon={Database}
        title="ClickHouse Connection Error"
        description={message || "Unable to connect to the ClickHouse database."}
        scope={ErrorScope.App}
      />
      <TroubleshootingSection scope={ErrorScope.App}>
        <>Verify ClickHouse is running and accessible</>
        <>
          Check the{" "}
          <ErrorInlineCode scope={ErrorScope.App}>
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
    <ErrorContentCard scope={ErrorScope.App}>
      <ErrorContentHeader
        icon={AlertTriangle}
        title={status ? `Error ${status}` : "Something Went Wrong"}
        description={message || "An unexpected error occurred."}
        scope={ErrorScope.App}
      />
      {stack && <StackTraceContent stack={stack} scope={ErrorScope.App} />}
    </ErrorContentCard>
  );
}
