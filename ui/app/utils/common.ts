import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type z from "zod";

/**
 * A utility to check if the current environment is a browser. Type-checking
 * `window` is not sufficient in some server runtimes (Deno, some test
 * environments, etc.) so we use the more robust check that is consistent with
 * React's internal logic.
 */
export const canUseDOM = !!(
  typeof window !== "undefined" &&
  window.document &&
  window.document.createElement
);

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

export function safeParseInt(
  value: string | number | null | undefined,
  defaultValue: number,
): number;

export function safeParseInt(
  value: string | number | null | undefined,
  defaultValue?: number,
): number | null;

export function safeParseInt(
  value: string | number | null | undefined,
  defaultValue?: number,
) {
  if (value === null || value === undefined) {
    return defaultValue ?? null;
  }

  const integer = Number.parseInt(value.toString(), 10);
  if (Number.isNaN(integer)) {
    return defaultValue ?? null;
  }

  return Number.isNaN(integer) ? (defaultValue ?? null) : integer;
}

/** @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set/isSupersetOf */
export function isSupersetOf(setA: Set<unknown>, setB: Set<unknown>): boolean {
  // @ts-expect-error: Check if native Set.prototype.isSuperset is available
  if (typeof Set.prototype.isSupersetOf === "function") {
    // @ts-expect-error: Call native
    return setA.isSupersetOf(setB);
  }

  for (const item of setB) {
    if (!setA.has(item)) {
      return false;
    }
  }

  return true;
}

/** @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set/isSubsetOf */
export function isSubsetOf(setA: Set<unknown>, setB: Set<unknown>): boolean {
  // @ts-expect-error: Check if native Set.prototype.isSubsetOf is available
  if (typeof Set.prototype.isSubsetOf === "function") {
    // @ts-expect-error: Call native
    return setA.isSubsetOf(setB);
  }

  for (const item of setA) {
    if (!setB.has(item)) {
      return false;
    }
  }

  return true;
}

/** @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set/symmetricDifference */
export function symmetricDifference<A, B = A>(
  setA: Set<A>,
  setB: Set<B>,
): Set<A | B> {
  // @ts-expect-error: Check if native Set.prototype.symmetricDifference is available
  if (typeof Set.prototype.symmetricDifference === "function") {
    // @ts-expect-error: Call native
    return setA.symmetricDifference(setB);
  }

  const result = new Set<A | B>();
  for (const item of setA) {
    if (!setB.has(item as unknown as B)) {
      result.add(item);
    }
  }
  for (const item of setB) {
    if (!setA.has(item as unknown as A)) {
      result.add(item);
    }
  }

  return result;
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

export class JSONParseError extends SyntaxError {
  constructor(
    public message: string,
    public cause?: unknown,
  ) {
    super(message, { cause });
    this.name = "JSONParseError";
  }
}

// Should be used for every Zod schema that matches a rust type
// for a bidirectional assertion that the types match.
// Returns the original schema so you can chain methods like .extend()
export function assertSchemaType<T>() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return <S extends z.ZodType<T, any, T>>(schema: S): S => {
    // Type assertion to ensure T compatibility at compile time
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const _typeCheck: z.ZodType<T, any, T> = schema;
    void _typeCheck;
    return schema;
  };
}
