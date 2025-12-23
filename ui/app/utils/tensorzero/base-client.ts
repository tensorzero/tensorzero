import { GatewayConnectionError, TensorZeroServerError } from "./errors";

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
  constructor(baseUrl: string, apiKey?: string | null) {
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

  protected async getErrorText(response: Response): Promise<string> {
    if (response.bodyUsed) {
      response = response.clone();
    }
    const responseText = await response.text();
    try {
      const parsed = JSON.parse(responseText);
      return typeof parsed?.error === "string" ? parsed.error : responseText;
    } catch {
      // Invalid JSON; return plain text from response
      return responseText;
    }
  }

  protected handleHttpError({
    message,
    response,
  }: {
    message: string;
    response: Response;
  }): never {
    throw new TensorZeroServerError(message, {
      // TODO: Ensure that server errors do not leak sensitive information to
      // the client before exposing the statusText
      // statusText: response.statusText,
      status: response.status,
    });
  }
}
