import { describe, it, expect } from "vitest";
import {
  InfraErrorType,
  GatewayConnectionError,
  isInfraErrorData,
  isAuthenticationError,
  isClickHouseError,
  isGatewayConnectionError,
  isGatewayRouteNotFoundError,
  TensorZeroServerError,
  classifyError,
  getErrorLabel,
} from "./errors";

describe("InfraErrorType", () => {
  it("should have expected error type values", () => {
    expect(InfraErrorType.GatewayUnavailable).toBe("GATEWAY_UNAVAILABLE");
    expect(InfraErrorType.GatewayAuthFailed).toBe("GATEWAY_AUTH_FAILED");
    expect(InfraErrorType.GatewayRouteNotFound).toBe("GATEWAY_ROUTE_NOT_FOUND");
    expect(InfraErrorType.ClickHouseUnavailable).toBe("CLICKHOUSE_UNAVAILABLE");
    expect(InfraErrorType.ServerError).toBe("SERVER_ERROR");
  });
});

describe("isInfraErrorData", () => {
  it("should return true for valid InfraErrorData", () => {
    const data = { errorType: InfraErrorType.GatewayUnavailable };
    expect(isInfraErrorData(data)).toBe(true);
  });

  it("should return true for InfraErrorData with optional fields", () => {
    const data = {
      errorType: InfraErrorType.GatewayRouteNotFound,
      message: "Some message",
      routeInfo: "GET /api/test",
    };
    expect(isInfraErrorData(data)).toBe(true);
  });

  it("should return false for null", () => {
    expect(isInfraErrorData(null)).toBe(false);
  });

  it("should return false for undefined", () => {
    expect(isInfraErrorData(undefined)).toBe(false);
  });

  it("should return false for object without errorType", () => {
    expect(isInfraErrorData({ message: "error" })).toBe(false);
  });

  it("should return false for object with invalid errorType", () => {
    expect(isInfraErrorData({ errorType: "INVALID_TYPE" })).toBe(false);
  });

  it("should return false for non-object values", () => {
    expect(isInfraErrorData("string")).toBe(false);
    expect(isInfraErrorData(123)).toBe(false);
    expect(isInfraErrorData(true)).toBe(false);
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

describe("isGatewayRouteNotFoundError", () => {
  it("should return true for TensorZeroServerError.RouteNotFound instance", () => {
    const error = new TensorZeroServerError.RouteNotFound(
      "Route not found: GET /api/unknown",
    );
    expect(isGatewayRouteNotFoundError(error)).toBe(true);
  });

  it("should return true for error with matching message pattern", () => {
    // Errors get serialized across React Router boundary, losing instanceof
    const error = new Error("Route not found: GET /api/unknown");
    expect(isGatewayRouteNotFoundError(error)).toBe(true);
  });

  it("should return true for plain object with matching message", () => {
    const serializedError = { message: "Route not found: POST /api/test" };
    expect(isGatewayRouteNotFoundError(serializedError)).toBe(true);
  });

  it("should return false for error with non-matching message", () => {
    const error = new Error("Something went wrong");
    expect(isGatewayRouteNotFoundError(error)).toBe(false);
  });

  it("should return false for error with partial message match", () => {
    const error = new Error("Cannot route not found");
    expect(isGatewayRouteNotFoundError(error)).toBe(false);
  });

  it("should return false for null", () => {
    expect(isGatewayRouteNotFoundError(null)).toBe(false);
  });

  it("should return false for undefined", () => {
    expect(isGatewayRouteNotFoundError(undefined)).toBe(false);
  });

  it("should return false for string", () => {
    expect(isGatewayRouteNotFoundError("Route not found: GET /api")).toBe(
      false,
    );
  });

  it("should return false for object without message property", () => {
    expect(isGatewayRouteNotFoundError({ error: "Route not found" })).toBe(
      false,
    );
  });

  it("should return false for object with non-string message", () => {
    expect(isGatewayRouteNotFoundError({ message: 404 })).toBe(false);
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

describe("classifyError", () => {
  it("should classify GatewayConnectionError as GatewayUnavailable", () => {
    const error = new GatewayConnectionError();
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.GatewayUnavailable);
  });

  it("should classify serialized GatewayConnectionError as GatewayUnavailable", () => {
    const serializedError = {
      name: "GatewayConnectionError",
      message: "Failed",
    };
    const result = classifyError(serializedError);
    expect(result.type).toBe(InfraErrorType.GatewayUnavailable);
  });

  it("should classify TensorZeroServerError with status 401 as GatewayAuthFailed", () => {
    const error = new TensorZeroServerError("Unauthorized", { status: 401 });
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.GatewayAuthFailed);
  });

  it("should classify RouteNotFound error as GatewayRouteNotFound with routeInfo", () => {
    const error = new TensorZeroServerError.RouteNotFound(
      "Route not found: GET /api/unknown",
    );
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.GatewayRouteNotFound);
    if (result.type === InfraErrorType.GatewayRouteNotFound) {
      expect(result.routeInfo).toBe("GET /api/unknown");
    }
  });

  it("should classify ClickHouse errors as ClickHouseUnavailable", () => {
    const error = new TensorZeroServerError.ClickHouseConnection(
      "Connection failed",
    );
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.ClickHouseUnavailable);
    if (result.type === InfraErrorType.ClickHouseUnavailable) {
      expect(result.message).toBe("Connection failed");
    }
  });

  it("should classify generic Error as ServerError", () => {
    const error = new Error("Something went wrong");
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.ServerError);
    if (result.type === InfraErrorType.ServerError) {
      expect(result.message).toBe("Something went wrong");
    }
  });

  it("should classify TensorZeroServerError with 500 status as ServerError", () => {
    const error = new TensorZeroServerError("Internal error", { status: 500 });
    const result = classifyError(error);
    expect(result.type).toBe(InfraErrorType.ServerError);
  });

  it("should classify null as ServerError", () => {
    const result = classifyError(null);
    expect(result.type).toBe(InfraErrorType.ServerError);
  });

  it("should classify undefined as ServerError", () => {
    const result = classifyError(undefined);
    expect(result.type).toBe(InfraErrorType.ServerError);
  });
});

describe("getErrorLabel", () => {
  it("should return correct label for GatewayUnavailable", () => {
    expect(getErrorLabel(InfraErrorType.GatewayUnavailable)).toBe(
      "Gateway Connection Error",
    );
  });

  it("should return correct label for GatewayAuthFailed", () => {
    expect(getErrorLabel(InfraErrorType.GatewayAuthFailed)).toBe("Auth Error");
  });

  it("should return correct label for GatewayRouteNotFound", () => {
    expect(getErrorLabel(InfraErrorType.GatewayRouteNotFound)).toBe(
      "Route Error",
    );
  });

  it("should return correct label for ClickHouseUnavailable", () => {
    expect(getErrorLabel(InfraErrorType.ClickHouseUnavailable)).toBe(
      "Database Error",
    );
  });

  it("should return correct label for ServerError", () => {
    expect(getErrorLabel(InfraErrorType.ServerError)).toBe("Server Error");
  });
});
