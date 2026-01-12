import { GatewayConnectionError, TensorZeroServerError } from "./errors";

/**
 * Parsed error response from the gateway.
 */
interface GatewayErrorResponse {
  /** Human-readable error message */
  message: string;
  /**
   * Error code from error_json discriminant (e.g., "RouteNotFound").
   * Only present when gateway has unstable_error_json enabled.
   */
  errorCode: string | null;
}

/**
 * Base client for TensorZero Gateway with shared infrastructure.
 * Provides common functionality for authentication, HTTP requests, and error handling.
 */
export class BaseTensorZeroClient {
  protected baseUrl: string;
  protected apiKey: string | null;

  /**
   * @param baseUrl - The base URL of the TensorZero Gateway (e.g. "http://localhost:3000")
   * @param apiKey - Optional API key for bearer authentication
   */
  constructor(baseUrl: string, apiKey?: string) {
    // Remove any trailing slash for consistency.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey ?? null;
  }

  protected async fetch(
    path: string,
    init: {
      method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
      body?: BodyInit;
      headers?: HeadersInit;
    },
  ) {
    const { method } = init;
    const url = `${this.baseUrl}${path}`;

    // For methods which expect payloads, always pass a body value even when it
    // is empty to deal with consistency issues in various runtimes.
    const expectsPayload =
      method === "POST" || method === "PUT" || method === "PATCH";
    const body = init.body || (expectsPayload ? "" : undefined);
    const headers = new Headers(init.headers);
    if (!headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }

    // Add bearer auth for all endpoints except /status
    if (this.apiKey && path !== "/status") {
      headers.set("authorization", `Bearer ${this.apiKey}`);
    }

    try {
      return await fetch(url, { method, headers, body });
    } catch (error) {
      // Convert network errors (ECONNREFUSED, fetch failed, etc.) to GatewayConnectionError
      throw new GatewayConnectionError(error);
    }
  }

  /**
   * Parse error response from gateway, extracting message and errorCode.
   * The errorCode is extracted from error_json when the gateway has unstable_error_json enabled.
   */
  protected async parseErrorResponse(
    response: Response,
  ): Promise<GatewayErrorResponse> {
    if (response.bodyUsed) {
      response = response.clone();
    }
    const responseText = await response.text();
    try {
      const parsed = JSON.parse(responseText);
      const message =
        typeof parsed?.error === "string" ? parsed.error : responseText;

      // Extract errorCode from error_json if present (requires unstable_error_json in gateway config)
      // The error_json structure is: { "ErrorVariantName": { ...details } }
      let errorCode: string | null = null;
      if (
        parsed?.error_json &&
        typeof parsed.error_json === "object" &&
        !Array.isArray(parsed.error_json)
      ) {
        const keys = Object.keys(parsed.error_json);
        if (keys.length === 1) {
          errorCode = keys[0];
        }
      }

      return { message, errorCode };
    } catch {
      // Invalid JSON; return plain text from response
      return { message: responseText, errorCode: null };
    }
  }

  /**
   * @deprecated Use parseErrorResponse instead for structured error handling.
   */
  protected async getErrorText(response: Response): Promise<string> {
    const { message } = await this.parseErrorResponse(response);
    return message;
  }

  protected handleHttpError({
    message,
    response,
    errorCode,
  }: {
    message: string;
    response: Response;
    errorCode?: string | null;
  }): never {
    throw new TensorZeroServerError(message, {
      // TODO: Ensure that server errors do not leak sensitive information to
      // the client before exposing the statusText
      // statusText: response.statusText,
      status: response.status,
      errorCode: errorCode ?? undefined,
    });
  }
}
