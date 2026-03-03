import type { ActionFunctionArgs } from "react-router";
import { data } from "react-router";
import { getEnv } from "~/utils/env.server";
import { apiKeyCookie } from "~/utils/api-key-override.server";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const apiKey = formData.get("apiKey");

  if (typeof apiKey !== "string" || !apiKey.trim()) {
    return data({ error: "API key is required" }, { status: 400 });
  }

  const trimmed = apiKey.trim();

  if (trimmed.length > 512) {
    return data({ error: "API key is too long" }, { status: 400 });
  }
  const env = getEnv();

  if (env.TENSORZERO_API_KEY) {
    return data(
      {
        error:
          "`TENSORZERO_API_KEY` is already configured on the server. Update the environment variable instead.",
      },
      { status: 409 },
    );
  }

  // Validate the key by making a test request to the gateway
  const validateUrl = new URL(
    "/internal/ui_config",
    env.TENSORZERO_GATEWAY_URL,
  );
  try {
    const response = await fetch(validateUrl, {
      headers: {
        authorization: `Bearer ${trimmed}`,
      },
    });

    if (response.status === 401) {
      return data(
        { error: "Invalid API key. The gateway rejected this key." },
        { status: 401 },
      );
    }

    if (!response.ok) {
      return data(
        {
          error: `Gateway returned ${response.status}. Check that the gateway is running.`,
        },
        { status: 502 },
      );
    }
  } catch {
    return data(
      { error: "Unable to connect to the gateway. Check that it is running." },
      { status: 502 },
    );
  }

  // Key is valid — store it in an HTTP-only cookie on the user's browser.
  return data(
    { success: true },
    { headers: { "Set-Cookie": await apiKeyCookie.serialize(trimmed) } },
  );
}
