import { describe, it, expect } from "vitest";
import {
  BoundaryErrorType,
  GatewayConnectionError,
  isBoundaryErrorData,
  isAuthenticationError,
  isClickHouseError,
  isGatewayConnectionError,
  isRouteNotFoundError,
  TensorZeroServerError,
} from "./errors";

describe("BoundaryErrorType", () => {
  it("should have expected error type values", () => {
    expect(BoundaryErrorType.GatewayUnavailable).toBe("GATEWAY_UNAVAILABLE");
    expect(BoundaryErrorType.GatewayAuthFailed).toBe("GATEWAY_AUTH_FAILED");
    expect(BoundaryErrorType.RouteNotFound).toBe("ROUTE_NOT_FOUND");
    expect(BoundaryErrorType.ClickHouseConnection).toBe(
      "CLICKHOUSE_CONNECTION",
    );
    expect(BoundaryErrorType.ServerError).toBe("SERVER_ERROR");
  });
});

describe("isBoundaryErrorData", () => {
  it("should return true for valid BoundaryErrorData", () => {
    const data = { errorType: BoundaryErrorType.GatewayUnavailable };
    expect(isBoundaryErrorData(data)).toBe(true);
  });

  it("should return true for BoundaryErrorData with optional fields", () => {
    const data = {
      errorType: BoundaryErrorType.RouteNotFound,
      message: "Some message",
      routeInfo: "GET /api/test",
    };
    expect(isBoundaryErrorData(data)).toBe(true);
  });

  it("should return false for null", () => {
    expect(isBoundaryErrorData(null)).toBe(false);
  });

  it("should return false for undefined", () => {
    expect(isBoundaryErrorData(undefined)).toBe(false);
  });

  it("should return false for object without errorType", () => {
    expect(isBoundaryErrorData({ message: "error" })).toBe(false);
  });

  it("should return false for object with invalid errorType", () => {
    expect(isBoundaryErrorData({ errorType: "INVALID_TYPE" })).toBe(false);
  });

  it("should return false for non-object values", () => {
    expect(isBoundaryErrorData("string")).toBe(false);
    expect(isBoundaryErrorData(123)).toBe(false);
    expect(isBoundaryErrorData(true)).toBe(false);
  });
});

describe("isClickHouseError", () => {
  it("should return true for ClickHouseConnection error", () => {
    const error = new TensorZeroServerError.ClickHouseConnection(
      "Connection failed",
    );
    expect(isClickHouseError(error)).toBe(true);
  });

  it("should return true for ClickHouseQuery error", () => {
    const error = new TensorZeroServerError.ClickHouseQuery("Query failed");
    expect(isClickHouseError(error)).toBe(true);
  });

  it("should return true for ClickHouseDeserialization error", () => {
    const error = new TensorZeroServerError.ClickHouseDeserialization(
      "Deserialization failed",
    );
    expect(isClickHouseError(error)).toBe(true);
  });

  it("should return true for ClickHouseMigration error", () => {
    const error = new TensorZeroServerError.ClickHouseMigration(
      "Migration failed",
    );
    expect(isClickHouseError(error)).toBe(true);
  });

  it("should return true for serialized error with ClickHouse in name", () => {
    // Errors get serialized across React Router boundary, losing instanceof
    const serializedError = {
      name: "ClickHouseConnectionError",
      message: "Failed",
    };
    expect(isClickHouseError(serializedError)).toBe(true);
  });

  it("should return false for error that merely mentions ClickHouse in message", () => {
    // Message-based matching is intentionally not supported - too broad
    const error = new Error("Unable to connect to ClickHouse database");
    expect(isClickHouseError(error)).toBe(false);
  });

  it("should return false for unrelated errors", () => {
    const error = new Error("Something went wrong");
    expect(isClickHouseError(error)).toBe(false);
  });

  it("should return false for null and undefined", () => {
    expect(isClickHouseError(null)).toBe(false);
    expect(isClickHouseError(undefined)).toBe(false);
  });
});

describe("isRouteNotFoundError", () => {
  it("should return true for TensorZeroServerError.RouteNotFound instance", () => {
    const error = new TensorZeroServerError.RouteNotFound(
      "Route not found: GET /api/unknown",
    );
    expect(isRouteNotFoundError(error)).toBe(true);
  });

  it("should return true for error with matching message pattern", () => {
    // Errors get serialized across React Router boundary, losing instanceof
    const error = new Error("Route not found: GET /api/unknown");
    expect(isRouteNotFoundError(error)).toBe(true);
  });

  it("should return true for plain object with matching message", () => {
    const serializedError = { message: "Route not found: POST /api/test" };
    expect(isRouteNotFoundError(serializedError)).toBe(true);
  });

  it("should return false for error with non-matching message", () => {
    const error = new Error("Something went wrong");
    expect(isRouteNotFoundError(error)).toBe(false);
  });

  it("should return false for error with partial message match", () => {
    const error = new Error("Cannot route not found");
    expect(isRouteNotFoundError(error)).toBe(false);
  });

  it("should return false for null", () => {
    expect(isRouteNotFoundError(null)).toBe(false);
  });

  it("should return false for undefined", () => {
    expect(isRouteNotFoundError(undefined)).toBe(false);
  });

  it("should return false for string", () => {
    expect(isRouteNotFoundError("Route not found: GET /api")).toBe(false);
  });

  it("should return false for object without message property", () => {
    expect(isRouteNotFoundError({ error: "Route not found" })).toBe(false);
  });

  it("should return false for object with non-string message", () => {
    expect(isRouteNotFoundError({ message: 404 })).toBe(false);
  });
});

describe("GatewayConnectionError", () => {
  it("should create error with correct name and message", () => {
    const error = new GatewayConnectionError();
    expect(error.name).toBe("GatewayConnectionError");
    expect(error.message).toBe("Cannot connect to TensorZero Gateway");
  });

  it("should preserve cause when provided", () => {
    const cause = new Error("Network timeout");
    const error = new GatewayConnectionError(cause);
    expect(error.cause).toBe(cause);
  });

  it("should be instanceof Error", () => {
    const error = new GatewayConnectionError();
    expect(error instanceof Error).toBe(true);
  });
});

describe("isGatewayConnectionError", () => {
  it("should return true for GatewayConnectionError instance", () => {
    const error = new GatewayConnectionError();
    expect(isGatewayConnectionError(error)).toBe(true);
  });

  it("should return true for serialized error with matching name", () => {
    // Errors get serialized across React Router boundary, losing instanceof
    const serializedError = {
      name: "GatewayConnectionError",
      message: "Connection failed",
    };
    expect(isGatewayConnectionError(serializedError)).toBe(true);
  });

  it("should return false for other Error instances", () => {
    const error = new Error("Connection failed");
    expect(isGatewayConnectionError(error)).toBe(false);
  });

  it("should return false for TensorZeroServerError", () => {
    const error = new TensorZeroServerError("Server error");
    expect(isGatewayConnectionError(error)).toBe(false);
  });

  it("should return false for null and undefined", () => {
    expect(isGatewayConnectionError(null)).toBe(false);
    expect(isGatewayConnectionError(undefined)).toBe(false);
  });
});

describe("isAuthenticationError", () => {
  it("should return true for TensorZeroServerError with status 401", () => {
    const error = new TensorZeroServerError("Unauthorized", { status: 401 });
    expect(isAuthenticationError(error)).toBe(true);
  });

  it("should return true for serialized error with status 401", () => {
    // Errors get serialized across React Router boundary
    const serializedError = {
      name: "TensorZeroServerError",
      status: 401,
      message: "Unauthorized",
    };
    expect(isAuthenticationError(serializedError)).toBe(true);
  });

  it("should return false for TensorZeroServerError with other status", () => {
    const error = new TensorZeroServerError("Forbidden", { status: 403 });
    expect(isAuthenticationError(error)).toBe(false);
  });

  it("should return false for serialized error with other status", () => {
    const serializedError = {
      name: "TensorZeroServerError",
      status: 500,
      message: "Server error",
    };
    expect(isAuthenticationError(serializedError)).toBe(false);
  });

  it("should return false for regular Error", () => {
    const error = new Error("Unauthorized");
    expect(isAuthenticationError(error)).toBe(false);
  });

  it("should return false for GatewayConnectionError", () => {
    const error = new GatewayConnectionError();
    expect(isAuthenticationError(error)).toBe(false);
  });

  it("should return false for null and undefined", () => {
    expect(isAuthenticationError(null)).toBe(false);
    expect(isAuthenticationError(undefined)).toBe(false);
  });
});
