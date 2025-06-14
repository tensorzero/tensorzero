import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * A helper function to merge class names conditionally, and deduplicating potentially conflicting
 * Tailwind classes. Note that deduplication can have a potentially significant performance overhead
 * for already slow-rendering components, and it is generally only useful in cases where a component
 * defines its own classes and receives potentially conflicting classes via props. If all you need is
 * to conditionally apply classes, it is preferred to use `clsx` directly.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// extract Unix timestamp from a UUID v7 value without an external library
export function extractTimestampFromUUIDv7(uuid: string): Date {
  // split the UUID into its components
  const parts = uuid.split("-");

  // the second part of the UUID contains the high bits of the timestamp (48 bits in total)
  const highBitsHex = parts[0] + parts[1].slice(0, 4);

  // convert the high bits from hex to decimal
  // the UUID v7 timestamp is the number of milliseconds since Unix epoch (January 1, 1970)
  const timestampInMilliseconds = parseInt(highBitsHex, 16);

  // convert the timestamp to a Date object
  const date = new Date(timestampInMilliseconds);

  return date;
}

/**
 * Utility function to debounce a request for a certain number of milliseconds
 * in route client loaders.
 *
 * @see
 * https://programmingarehard.com/2025/02/24/debouncing-in-react-router-v7.html/
 */
export function abortableTimeout(request: Request, ms: number) {
  const { signal } = request;
  return new Promise((resolve, reject) => {
    // If the signal is aborted by the time it reaches this, reject
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }

    // Schedule the resolve function to be called in the future a certain number
    // of milliseconds
    const timeoutId = setTimeout(resolve, ms);

    // Listen for the abort event. If it fires, reject
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timeoutId);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

interface ErrorLike {
  message: string;
  name: string;
  stack?: string;
  cause?: unknown;
}

export function isErrorLike(error: unknown): error is ErrorLike {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string" &&
    "name" in error &&
    typeof error.name === "string" &&
    ("stack" in error ? typeof error.stack === "string" : true)
  );
}

export class JSONParseError extends SyntaxError {}

export class ServerRequestError extends Error {
  statusCode: number;
  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "ServerRequestError";
    this.statusCode = statusCode;
  }
}

export function isServerRequestError(
  error: unknown,
): error is ServerRequestError {
  return isErrorLike(error) && error.name === "ServerRequestError";
}
