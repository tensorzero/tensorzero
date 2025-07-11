import type { LoaderFunctionArgs, ActionFunctionArgs } from "react-router";
import { data } from "react-router";
import { getEnv } from "~/utils/env.server";

async function proxyRequest(request: Request) {
  const url = URL.parse(request.url);
  if (!url) {
    return data({}, 400);
  }

  // Actual path comes after `/api/gateway`
  const path = url.pathname.replace(/^\/api\/gateway/, "");

  const gatewayUrl = new URL(
    path + url.search,
    getEnv().TENSORZERO_GATEWAY_URL,
  );

  const init: RequestInit = {
    method: request.method,
    headers: request.headers,
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    // @ts-expect-error - duplex is required for streaming bodies but not in TypeScript types
    init.duplex = "half";
  }

  return fetch(gatewayUrl, init);
}

// GET
export async function loader({ request }: LoaderFunctionArgs) {
  return proxyRequest(request);
}

// POST, PUT, PATCH, DELETE, etc.
export async function action({ request }: ActionFunctionArgs) {
  return proxyRequest(request);
}
