import { createRequire } from "module";
import type { EditPayload, KeyInfo } from "./bindings";
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
