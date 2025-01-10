import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

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
