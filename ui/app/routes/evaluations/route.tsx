import type { Route } from "./+types/route";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const run_ids = searchParams.get("run_ids");
  const run_ids_array = run_ids ? run_ids.split(",") : [];
}
