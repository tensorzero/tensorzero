import { describe, expect, it } from "vitest";
import { buildGatewayUrl } from "./gateway-url";

describe("buildGatewayUrl", () => {
  it("preserves a configured base path without a trailing slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1",
        "/internal/autopilot/v1/sessions/session-1/events/stream",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/autopilot/v1/sessions/session-1/events/stream",
    );
  });

  it("preserves a configured base path with a trailing slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "/internal/autopilot/v1/sessions/session-1/events/stream",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/autopilot/v1/sessions/session-1/events/stream",
    );
  });

  it("works when the provided path omits the leading slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "internal/ui_config",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/ui_config",
    );
  });

  it("preserves query strings on the provided path", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "/internal/autopilot/v1/sessions/session-1/events?before=cursor-1&limit=21",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/autopilot/v1/sessions/session-1/events?before=cursor-1&limit=21",
    );
  });

  it("still works for a gateway configured at the origin root", () => {
    expect(
      buildGatewayUrl("http://tensorzero-gateway:3000/", "/internal/ui_config")
        .toString(),
    ).toBe("http://tensorzero-gateway:3000/internal/ui_config");
  });
});
