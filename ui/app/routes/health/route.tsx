export async function loader() {
  // Health check does not verify gateway or database connectivity.
  return new Response(null, { status: 200 });
}
