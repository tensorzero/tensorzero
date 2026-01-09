import { StatusCodes as HttpStatusCode } from "http-status-codes";
import { isRouteErrorResponse } from "react-router";
import { isErrorLike } from "~/utils/common";

/**
 * Enum-like constants for categorizing infrastructure errors that survive
 * serialization across the React Router server/client boundary.
 *
 * Use with React Router's `data()` helper:
 * ```ts
 * throw data({ errorType: InfraErrorType.GatewayUnavailable }, { status: 503 });
 * ```
 */
export const InfraErrorType = {
  GatewayUnavailable: "GATEWAY_UNAVAILABLE",
  GatewayAuthFailed: "GATEWAY_AUTH_FAILED",
  GatewayRouteNotFound: "GATEWAY_ROUTE_NOT_FOUND",
  ClickHouseUnavailable: "CLICKHOUSE_UNAVAILABLE",
  ServerError: "SERVER_ERROR",
} as const;

export type InfraErrorType =
  (typeof InfraErrorType)[keyof typeof InfraErrorType];

/**
 * Discriminated union for error data passed via React Router's `data()` helper.
 * Each variant only includes fields relevant to that error type, enforcing
 * valid combinations at compile time.
 */
export type InfraErrorData =
  | { errorType: typeof InfraErrorType.GatewayUnavailable }
  | { errorType: typeof InfraErrorType.GatewayAuthFailed }
  | { errorType: typeof InfraErrorType.GatewayRouteNotFound; routeInfo: string }
  | {
      errorType: typeof InfraErrorType.ClickHouseUnavailable;
      message?: string;
    }
  | { errorType: typeof InfraErrorType.ServerError; message?: string };

/**
 * Discriminated union for classified errors used in error rendering.
 * Mirrors InfraErrorData but uses 'type' for consistency with component props,
 * and includes additional fields like 'status' for HTTP status codes.
 */
export type ClassifiedError =
  | { type: typeof InfraErrorType.GatewayUnavailable }
  | { type: typeof InfraErrorType.GatewayAuthFailed }
  | { type: typeof InfraErrorType.GatewayRouteNotFound; routeInfo: string }
  | { type: typeof InfraErrorType.ClickHouseUnavailable; message?: string }
  | {
      type: typeof InfraErrorType.ServerError;
      message?: string;
      status?: number;
      stack?: string;
    };

/**
 * Type guard to check if a value is InfraErrorData.
 */
export function isInfraErrorData(value: unknown): value is InfraErrorData {
  return (
    typeof value === "object" &&
    value !== null &&
    "errorType" in value &&
    typeof value.errorType === "string" &&
    Object.values(InfraErrorType).includes(value.errorType as InfraErrorType)
  );
}

/**
 * Error thrown when the UI cannot connect to the TensorZero gateway.
 * This is distinct from TensorZeroServerError which represents errors
 * returned by the gateway itself.
 */
export class GatewayConnectionError extends Error {
  constructor(cause?: unknown) {
    super("Cannot connect to TensorZero Gateway", { cause });
    this.name = "GatewayConnectionError";
  }
}

export function isGatewayConnectionError(
  error: unknown,
): error is GatewayConnectionError {
  // Check instanceof for server-side errors
  if (error instanceof GatewayConnectionError) {
    return true;
  }
  // Check serialized object properties (works if thrown from server loader)
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "GatewayConnectionError"
  );
}

export function isAuthenticationError(error: unknown): boolean {
  if (error instanceof TensorZeroServerError) {
    return error.status === 401;
  }
  // Check serialized object properties (works if thrown from server loader)
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "TensorZeroServerError" &&
    "status" in error &&
    error.status === 401
  );
}

/**
 * Check if an error indicates a gateway API route not found.
 * This typically happens when the UI version doesn't match the gateway version.
 *
 * Supports both:
 * - Direct TensorZeroServerError.RouteNotFound instances (server-side)
 * - Message pattern matching (for serialized errors)
 */
export function isGatewayRouteNotFoundError(error: unknown): boolean {
  if (error instanceof TensorZeroServerError.RouteNotFound) {
    return true;
  }

  // Check message pattern - this works even after serialization
  // The message format is: "Route not found: METHOD /path"
  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string" &&
    error.message.startsWith("Route not found:")
  ) {
    return true;
  }

  return false;
}

/**
 * Check if an error indicates a ClickHouse connection or query failure.
 *
 * Supports both:
 * - Direct TensorZeroServerError.ClickHouse* instances (server-side)
 * - Name pattern matching (for serialized errors where instanceof fails)
 */
export function isClickHouseError(error: unknown): boolean {
  // Direct instanceof checks for all ClickHouse error subclasses
  if (
    error instanceof TensorZeroServerError.ClickHouseConnection ||
    error instanceof TensorZeroServerError.ClickHouseQuery ||
    error instanceof TensorZeroServerError.ClickHouseDeserialization ||
    error instanceof TensorZeroServerError.ClickHouseMigration
  ) {
    return true;
  }

  // Check serialized error name - all ClickHouse errors have names prefixed with "ClickHouse"
  // This handles the serialization boundary where instanceof fails
  if (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    typeof error.name === "string" &&
    error.name.startsWith("ClickHouse")
  ) {
    return true;
  }

  return false;
}

/**
 * Check if an error is an infrastructure error that should trigger graceful degradation.
 * Includes: gateway unreachable, auth failed, route not found (version mismatch), ClickHouse unavailable.
 */
export function isInfraError(error: unknown): boolean {
  return (
    isGatewayConnectionError(error) ||
    isAuthenticationError(error) ||
    isGatewayRouteNotFoundError(error) ||
    isClickHouseError(error)
  );
}

/**
 * Check if an error indicates that Autopilot is not configured.
 * The gateway returns 501 NOT_IMPLEMENTED when autopilot is not implemented,
 * or 401 UNAUTHORIZED when the autopilot API key is not configured.
 */
export function isAutopilotUnavailableError(error: unknown): boolean {
  if (error instanceof TensorZeroServerError) {
    return error.status === 501 || error.status === 401;
  }
  // Check serialized object properties (works if thrown from server loader)
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "TensorZeroServerError" &&
    "status" in error &&
    (error.status === 501 || error.status === 401)
  );
}

export class TensorZeroServerError extends Error {
  readonly status: number;
  readonly statusText: string | null;
  constructor(
    error: unknown,
    args?: {
      status:
        | HttpStatusCode
        | keyof typeof HttpStatusCode
        | (typeof HttpStatusCode)[keyof typeof HttpStatusCode];
      statusText?: string;
    },
  ) {
    if (isErrorLike(error)) {
      super(error.message, { cause: error.cause });
    } else {
      super(typeof error === "string" ? error : "Unknown server error");
    }

    const { status = HttpStatusCode.INTERNAL_SERVER_ERROR, statusText = null } =
      args ?? {};
    this.name = "TensorZeroServerError";
    this.status = typeof status === "string" ? HttpStatusCode[status] : status;
    this.statusText = statusText;
  }

  // TODO: These are all copied from error.rs internal errors since we want to
  // rely on these responses for more detailed error handling in the frontend.
  // These should be generated or otherwise synchronized with the backend
  // without manual copying.
  static AllVariantsFailed = class AllVariantsFailed extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_GATEWAY });
    }
  };
  static ApiKeyMissing = class ApiKeyMissing extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static ExtraBodyReplacement = class ExtraBodyReplacement extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static AppState = class AppState extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static BadCredentialsPreInference = class BadCredentialsPreInference extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static BatchInputValidation = class BatchInputValidation extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static BatchNotFound = class BatchNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static Cache = class Cache extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ChannelWrite = class ChannelWrite extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ClickHouseConnection = class ClickHouseConnection extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ClickHouseDeserialization = class ClickHouseDeserialization extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ClickHouseMigration = class ClickHouseMigration extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ClickHouseQuery = class ClickHouseQuery extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ObjectStoreUnconfigured = class ObjectStoreUnconfigured extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static DatapointNotFound = class DatapointNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static Config = class Config extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static DynamicJsonSchema = class DynamicJsonSchema extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static FileRead = class FileRead extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static GCPCredentials = class GCPCredentials extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidInferenceTarget = class InvalidInferenceTarget extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static Inference = class Inference extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ObjectStoreWrite = class ObjectStoreWrite extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InferenceClient = class InferenceClient extends TensorZeroServerError {
    constructor(
      message: string,
      args?: {
        status?:
          | HttpStatusCode
          | (typeof HttpStatusCode)[keyof typeof HttpStatusCode];
      },
    ) {
      super(message, {
        status: args?.status ?? HttpStatusCode.INTERNAL_SERVER_ERROR,
      });
    }
  };
  static BadImageFetch = class BadImageFetch extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InferenceNotFound = class InferenceNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static InferenceServer = class InferenceServer extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InferenceTimeout = class InferenceTimeout extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.REQUEST_TIMEOUT });
    }
  };
  static ModelProviderTimeout = class ModelProviderTimeout extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.REQUEST_TIMEOUT });
    }
  };
  static VariantTimeout = class VariantTimeout extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.REQUEST_TIMEOUT });
    }
  };
  static InvalidClientMode = class InvalidClientMode extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidTensorzeroUuid = class InvalidTensorzeroUuid extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidUuid = class InvalidUuid extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InputValidation = class InputValidation extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InternalError = class InternalError extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidBaseUrl = class InvalidBaseUrl extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnsupportedContentBlockType = class UnsupportedContentBlockType extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidBatchParams = class InvalidBatchParams extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidCandidate = class InvalidCandidate extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidDiclConfig = class InvalidDiclConfig extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidDatasetName = class InvalidDatasetName extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidWorkflowEvaluationRun = class InvalidWorkflowEvaluationRun extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidFunctionVariants = class InvalidFunctionVariants extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidInferenceOutputSource = class InvalidInferenceOutputSource extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidMessage = class InvalidMessage extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidMetricName = class InvalidMetricName extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidModel = class InvalidModel extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidModelProvider = class InvalidModelProvider extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidOpenAICompatibleRequest = class InvalidOpenAICompatibleRequest extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidProviderConfig = class InvalidProviderConfig extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidRequest = class InvalidRequest extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static InvalidTemplatePath = class InvalidTemplatePath extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidTool = class InvalidTool extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static InvalidVariantForOptimization = class InvalidVariantForOptimization extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static JsonRequest = class JsonRequest extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static JsonSchema = class JsonSchema extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static JsonSchemaValidation = class JsonSchemaValidation extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static MiniJinjaEnvironment = class MiniJinjaEnvironment extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static MiniJinjaTemplate = class MiniJinjaTemplate extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static MiniJinjaTemplateMissing = class MiniJinjaTemplateMissing extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static MiniJinjaTemplateRender = class MiniJinjaTemplateRender extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static MissingBatchInferenceResponse = class MissingBatchInferenceResponse extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static MissingFunctionInVariants = class MissingFunctionInVariants extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static MissingFileExtension = class MissingFileExtension extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static ModelProvidersExhausted = class ModelProvidersExhausted extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ModelValidation = class ModelValidation extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static Observability = class Observability extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static OutputParsing = class OutputParsing extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static OutputValidation = class OutputValidation extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ProviderNotFound = class ProviderNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static Serialization = class Serialization extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static StreamError = class StreamError extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static ToolNotFound = class ToolNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static ToolNotLoaded = class ToolNotLoaded extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static TypeConversion = class TypeConversion extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnknownCandidate = class UnknownCandidate extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnknownFunction = class UnknownFunction extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static UnknownModel = class UnknownModel extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnknownTool = class UnknownTool extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnknownVariant = class UnknownVariant extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static UnknownMetric = class UnknownMetric extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static UnsupportedFileExtension = class UnsupportedFileExtension extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static UnsupportedModelProviderForBatchInference = class UnsupportedModelProviderForBatchInference extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnsupportedVariantForBatchInference = class UnsupportedVariantForBatchInference extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static UnsupportedVariantForStreamingInference = class UnsupportedVariantForStreamingInference extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UnsupportedVariantForFunctionType = class UnsupportedVariantForFunctionType extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.INTERNAL_SERVER_ERROR });
    }
  };
  static UuidInFuture = class UuidInFuture extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.BAD_REQUEST });
    }
  };
  static RouteNotFound = class RouteNotFound extends TensorZeroServerError {
    constructor(message: string) {
      super(message, { status: HttpStatusCode.NOT_FOUND });
    }
  };
  static NativeClient = class NativeClient extends TensorZeroServerError {};
}

export function isTensorZeroServerError(
  error: unknown,
): error is TensorZeroServerError {
  return isErrorLike(error) && error.name === "TensorZeroServerError";
}

/**
 * Returns a user-friendly label for an InfraErrorType.
 * Used in error dialogs and UI components.
 */
export function getErrorLabel(type: InfraErrorType): string {
  switch (type) {
    case InfraErrorType.GatewayUnavailable:
      return "Gateway Connection Error";
    case InfraErrorType.GatewayAuthFailed:
      return "Auth Error";
    case InfraErrorType.GatewayRouteNotFound:
      return "Route Error";
    case InfraErrorType.ClickHouseUnavailable:
      return "Database Error";
    case InfraErrorType.ServerError:
      return "Server Error";
    default: {
      const _exhaustiveCheck: never = type;
      return "Server Error";
    }
  }
}

/**
 * Extracts error details from an unknown error in a type-safe way.
 * Useful for displaying error information in UI components.
 */
export function getErrorDetails(error: unknown): {
  message: string;
  status?: number;
} {
  if (error instanceof TensorZeroServerError) {
    return { message: error.message, status: error.status };
  }
  if (error instanceof GatewayConnectionError) {
    return { message: error.message };
  }
  if (error instanceof Error) {
    return { message: error.message };
  }
  return { message: String(error) };
}

function extractErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string"
  ) {
    return error.message;
  }
  return "";
}

/**
 * Classifies an error into a ClassifiedError discriminated union.
 * Handles both direct Error instances and serialized error data from React Router boundaries.
 */
export function classifyError(error: unknown): ClassifiedError {
  if (isRouteErrorResponse(error) && isInfraErrorData(error.data)) {
    const { errorType } = error.data;
    switch (errorType) {
      case InfraErrorType.GatewayUnavailable:
        return { type: InfraErrorType.GatewayUnavailable };
      case InfraErrorType.GatewayAuthFailed:
        return { type: InfraErrorType.GatewayAuthFailed };
      case InfraErrorType.GatewayRouteNotFound:
        return {
          type: InfraErrorType.GatewayRouteNotFound,
          routeInfo: error.data.routeInfo,
        };
      case InfraErrorType.ClickHouseUnavailable:
        return {
          type: InfraErrorType.ClickHouseUnavailable,
          message: "message" in error.data ? error.data.message : undefined,
        };
      case InfraErrorType.ServerError:
        return {
          type: InfraErrorType.ServerError,
          message: "message" in error.data ? error.data.message : undefined,
          status: error.status,
        };
      default: {
        const _exhaustiveCheck: never = errorType;
        return { type: InfraErrorType.ServerError, status: error.status };
      }
    }
  }

  if (isGatewayConnectionError(error)) {
    return { type: InfraErrorType.GatewayUnavailable };
  }

  if (isAuthenticationError(error)) {
    return { type: InfraErrorType.GatewayAuthFailed };
  }

  if (isGatewayRouteNotFoundError(error)) {
    const errorMessage = extractErrorMessage(error);
    const routeMatch = errorMessage.match(/Route not found: (\w+) (.+)/);
    const routeInfo = routeMatch
      ? `${routeMatch[1]} ${routeMatch[2]}`
      : errorMessage;
    return { type: InfraErrorType.GatewayRouteNotFound, routeInfo };
  }

  if (isClickHouseError(error)) {
    const message = error instanceof Error ? error.message : undefined;
    return { type: InfraErrorType.ClickHouseUnavailable, message };
  }

  const message = isRouteErrorResponse(error)
    ? error.statusText || undefined
    : error instanceof Error
      ? error.message
      : undefined;
  const status = isRouteErrorResponse(error) ? error.status : undefined;
  return { type: InfraErrorType.ServerError, message, status };
}
