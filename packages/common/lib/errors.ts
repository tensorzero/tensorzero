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
