import { createRequire } from "module";
import type {
  EditPayload,
  GatewayEvent,
  GatewayListConfigWritesResponse,
  KeyInfo,
  ListConfigWritesParams,
} from "./bindings";
import type {
  ConfigWriter as NativeConfigWriterType,
  PostgresClient as NativePostgresClientType,
} from "../index";

// Re-export types from bindings
export type * from "./bindings";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);

const {
  ConfigWriter: NativeConfigWriter,
  PostgresClient: NativePostgresClient,
} = require("../index.cjs") as typeof import("../index");

/**
 * Wrapper class for type safety and convenience
 * around the native PostgresClient
 */
export class PostgresClient {
  private nativePostgresClient: NativePostgresClientType;

  constructor(client: NativePostgresClientType) {
    this.nativePostgresClient = client;
  }

  static async fromPostgresUrl(url: string): Promise<PostgresClient> {
    return new PostgresClient(await NativePostgresClient.fromPostgresUrl(url));
  }

  async createApiKey(description?: string | null): Promise<string> {
    return this.nativePostgresClient.createApiKey(description);
  }

  async listApiKeys(limit?: number, offset?: number): Promise<KeyInfo[]> {
    const result = await this.nativePostgresClient.listApiKeys(limit, offset);
    return JSON.parse(result) as KeyInfo[];
  }

  async disableApiKey(publicId: string): Promise<string> {
    return this.nativePostgresClient.disableApiKey(publicId);
  }

  async updateApiKeyDescription(
    publicId: string,
    description?: string | null,
  ): Promise<KeyInfo> {
    const result = await this.nativePostgresClient.updateApiKeyDescription(
      publicId,
      description ?? null,
    );
    return JSON.parse(result) as KeyInfo;
  }
}

/**
 * Wrapper class for type safety and convenience
 * around the native ConfigWriter
 */
export class ConfigWriter {
  private nativeConfigWriter: NativeConfigWriterType;

  private constructor(writer: NativeConfigWriterType) {
    this.nativeConfigWriter = writer;
  }

  static async new(globPattern: string): Promise<ConfigWriter> {
    return new ConfigWriter(await NativeConfigWriter.new(globPattern));
  }

  async applyEdit(edit: EditPayload): Promise<string[]> {
    return this.nativeConfigWriter.applyEdit(JSON.stringify(edit));
  }
}

/**
 * Options for fetching config writes from the gateway API.
 */
export interface ListConfigWritesOptions {
  /** Base URL of the TensorZero gateway (e.g., "http://localhost:3000") */
  baseUrl: string;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Optional pagination parameters */
  params?: ListConfigWritesParams;
}

/**
 * Fetches config writes (write_config tool calls) for a session from the gateway API.
 *
 * @param sessionId - The session ID to fetch config writes for
 * @param options - Options including baseUrl, apiKey, and pagination params
 * @returns The list of config writes as GatewayEvents
 */
export async function listConfigWrites(
  sessionId: string,
  options: ListConfigWritesOptions,
): Promise<GatewayListConfigWritesResponse> {
  const { baseUrl, apiKey, params } = options;

  const searchParams = new URLSearchParams();
  if (params?.limit !== undefined) {
    searchParams.set("limit", params.limit.toString());
  }
  if (params?.offset !== undefined) {
    searchParams.set("offset", params.offset.toString());
  }

  const queryString = searchParams.toString();
  const url = `${baseUrl}/internal/autopilot/v1/sessions/${encodeURIComponent(sessionId)}/config-writes${queryString ? `?${queryString}` : ""}`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const response = await fetch(url, { method: "GET", headers });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Failed to fetch config writes: ${response.status} ${response.statusText} - ${errorText}`,
    );
  }

  return (await response.json()) as GatewayListConfigWritesResponse;
}

/**
 * Extracts the EditPayload from a config write event.
 *
 * @param event - A GatewayEvent that should be a write_config tool call
 * @returns The EditPayload from the event's arguments
 * @throws Error if the event is not a write_config tool call
 */
export function extractEditPayloadFromConfigWrite(
  event: GatewayEvent,
): EditPayload {
  if (event.payload.type !== "tool_call") {
    throw new Error(
      `Expected tool_call event but got ${event.payload.type} for event ${event.id}`,
    );
  }

  if (event.payload.name !== "write_config") {
    throw new Error(
      `Expected write_config tool call but got ${event.payload.name} for event ${event.id}`,
    );
  }

  return event.payload.arguments as EditPayload;
}

/**
 * Result of writing a config write to file.
 */
export interface WriteConfigWriteResult {
  /** The event ID that was processed */
  eventId: string;
  /** Paths of files that were written */
  writtenPaths: string[];
}

/**
 * Writes a single config write event to file using the ConfigWriter.
 *
 * @param configWriter - The ConfigWriter instance to use
 * @param event - The config write event to apply
 * @returns The paths of files that were written
 */
export async function writeConfigWriteToFile(
  configWriter: ConfigWriter,
  event: GatewayEvent,
): Promise<WriteConfigWriteResult> {
  const editPayload = extractEditPayloadFromConfigWrite(event);
  const writtenPaths = await configWriter.applyEdit(editPayload);
  return {
    eventId: event.id,
    writtenPaths,
  };
}

/**
 * Result of writing all config writes from a session.
 */
export interface WriteSessionConfigWritesResult {
  /** Results for each config write event */
  results: WriteConfigWriteResult[];
  /** Total number of config writes processed */
  totalProcessed: number;
}

/**
 * Fetches all config writes from a session and writes them to files.
 *
 * @param configWriter - The ConfigWriter instance to use
 * @param sessionId - The session ID to fetch config writes for
 * @param options - Options including baseUrl, apiKey, and pagination params
 * @returns Results containing paths of all files written
 */
export async function writeSessionConfigWritesToFile(
  configWriter: ConfigWriter,
  sessionId: string,
  options: ListConfigWritesOptions,
): Promise<WriteSessionConfigWritesResult> {
  const response = await listConfigWrites(sessionId, options);

  const results: WriteConfigWriteResult[] = [];
  for (const event of response.config_writes) {
    const result = await writeConfigWriteToFile(configWriter, event);
    results.push(result);
  }

  return {
    results,
    totalProcessed: results.length,
  };
}
