import { describe, it, expect } from "vitest";
import {
  BoundaryErrorType,
  isBoundaryErrorData,
  isClickHouseError,
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

  it("should return true for error with ClickHouse in name", () => {
    const error = { name: "ClickHouseConnectionError", message: "Failed" };
    expect(isClickHouseError(error)).toBe(true);
  });

  it("should return false for error that merely mentions clickhouse in message", () => {
    // Message-based matching is intentionally not supported - it's too broad
    // and could match unrelated errors that just mention "ClickHouse"
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
    const error = new Error("Route not found: GET /api/unknown");
    expect(isRouteNotFoundError(error)).toBe(true);
  });

  it("should return true for plain object with matching message", () => {
    const error = { message: "Route not found: POST /api/test" };
    expect(isRouteNotFoundError(error)).toBe(true);
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
