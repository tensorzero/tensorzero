import { createClient, type ClickHouseClient } from "@clickhouse/client";
import { canUseDOM, isErrorLike } from "../common";
import { getEnv } from "../env.server";

class ClickHouseClientError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ClickHouseClientError";
  }
}

let _clickhouseClient: ClickHouseClient | null = null;
export function getClickhouseClient(): ClickHouseClient {
  // Ensure this only runs on the server. Vite's React Router plugin should
  // ensure this is unreachable since the filename ends with `.server.ts`, but
  // this check adds additional assurance.
  if (canUseDOM) {
    throw new ClickHouseClientError(
      "clickhouseClient can only be used on the server side",
    );
  }

  if (_clickhouseClient) {
    return _clickhouseClient;
  }

  const env = getEnv();
  try {
    // Proxy the ClickHouse client to intercept method calls for better error
    // handling. The ClickHouse client itself does not handle some errors
    // internally so the user will see cryptic Node errors unless we intercept.
    const client = new Proxy(
      createClient({
        url: env.TENSORZERO_CLICKHOUSE_URL,
        request_timeout: 1000 * 60 * 5,
        clickhouse_settings: {
          join_algorithm: 'auto'
        }
      }),
      {
        get(target, prop, receiver) {
          const propertyOrMethod = target[prop as keyof ClickHouseClient];

          // Intercept the method to catch errors and throw ClickHouseClientError
          if (typeof propertyOrMethod === "function") {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return function (...args: any[]) {
              try {
                const result = Reflect.apply(propertyOrMethod, target, args);
                if (isPromiseLike(result)) {
                  return result.catch((error: unknown) => {
                    throw getMethodError(error, prop);
                  });
                }
                return result;
              } catch (error) {
                throw getMethodError(error, prop);
              }
            };
          }

          // Otherwise, just return the property
          return Reflect.get(target, prop, receiver);
        },
      },
    );

    _clickhouseClient = client;
    return client;
  } catch (error) {
    throw new ClickHouseClientError(
      "Failed to create ClickHouse client. Please ensure that the `TENSORZERO_CLICKHOUSE_URL` environment variable is set correctly and that the ClickHouse server is running.\n\n" +
        "Failed with the following message:\n\n" +
        (isErrorLike(error) ? error.message : String(error)),
      { cause: error },
    );
  }
}

function getMethodError(originalError: unknown, methodName: string | symbol) {
  const originalErrorMessage = isErrorLike(originalError)
    ? originalError.message
    : String(originalError);

  let errorMessage =
    `Failed to execute ClickHouse ${methodName.toString()}. Please ensure that:\n` +
    "\n  1. the `TENSORZERO_CLICKHOUSE_URL` environment variable is set to the correct URL," +
    "\n  2. the ClickHouse server is running, and" +
    "\n  3. the query is valid\n";

  if (originalErrorMessage) {
    errorMessage += `\nFailed with the following message:\n\n${originalErrorMessage}`;
  }

  return new ClickHouseClientError(errorMessage, { cause: originalError });
}

export async function checkClickHouseConnection(): Promise<boolean> {
  try {
    const result = await getClickhouseClient().ping();
    return result.success;
  } catch {
    return false;
  }
}

export function isClickHouseClientError(
  error: unknown,
): error is ClickHouseClientError {
  return isErrorLike(error) && error.name === "ClickHouseClientError";
}

function isPromiseLike<T>(value: unknown): value is Promise<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Promise<T>).then === "function" &&
    typeof (value as Promise<T>).catch === "function"
  );
}
