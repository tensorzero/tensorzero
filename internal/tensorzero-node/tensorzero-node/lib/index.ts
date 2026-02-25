import { createRequire } from "module";
import type { EditPayload, KeyInfo } from "./bindings";
import type {
  ConfigApplier as NativeConfigApplierType,
  PostgresClient as NativePostgresClientType,
} from "../index";

// Re-export types from bindings
export type * from "./bindings";

// Use createRequire to load CommonJS module
const require = createRequire(import.meta.url);

const {
  ConfigApplier: NativeConfigApplier,
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
 * around the native ConfigApplier
 */
export class ConfigApplier {
  private nativeConfigApplier: NativeConfigApplierType;

  private constructor(applier: NativeConfigApplierType) {
    this.nativeConfigApplier = applier;
  }

  static async new(globPattern: string): Promise<ConfigApplier> {
    return new ConfigApplier(await NativeConfigApplier.new(globPattern));
  }

  async applyEdit(edit: EditPayload): Promise<string[]> {
    return this.nativeConfigApplier.applyEdit(JSON.stringify(edit));
  }
}
