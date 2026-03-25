/**
 * Joins a configured gateway URL with an internal API path while preserving any
 * non-root base path in the configured URL.
 *
 * Accepts `baseUrl` with or without trailing slash, and accepts `path` with or without
 * leading slash.
 */
export function buildGatewayUrl(baseUrl: string, path: string): URL {
  const url = new URL(baseUrl);
  const [rawPathname, ...searchParts] = path.split("?");
  const normalizedBasePath =
    url.pathname === "/" ? "" : url.pathname.replace(/\/+$/, "");
  const normalizedPathname = rawPathname.replace(/^\/+/, "");

  url.pathname = normalizedPathname
    ? `${normalizedBasePath}/${normalizedPathname}`
    : normalizedBasePath || "/";
  url.search = searchParts.length > 0 ? `?${searchParts.join("?")}` : "";
  url.hash = "";

  return url;
}
