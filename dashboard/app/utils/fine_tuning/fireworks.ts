export const FIREWORKS_API_URL = "https://api.fireworks.ai";
export const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY || throwError();
export const FIREWORKS_ACCOUNT_ID =
  process.env.FIREWORKS_ACCOUNT_ID || throwError();

// This is apparently the traditional way to coerce both to strings.
function throwError(): never {
  throw new Error("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID must be set");
}
