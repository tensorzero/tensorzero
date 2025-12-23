import { StatusCodes as HttpStatusCode } from "http-status-codes";
import { isErrorLike } from "~/utils/common";

/**
 * Error thrown when the UI cannot connect to the TensorZero gateway.
 * This is distinct from TensorZeroServerError which represents errors
 * returned by the gateway itself.
 */
export class GatewayConnectionError extends Error {
  constructor(cause?: unknown) {
    super("Cannot connect to TensorZero gateway", { cause });
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
