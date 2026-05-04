/**
 * `/functions` — list active functions and provide a "New function" form.
 *
 * Calls `GET /internal/functions` for the table and `POST /internal/functions`
 * for the create action. Variant config is constructed inline as plain JSON
 * because the typed `UninitializedFunctionConfig` shape isn't currently
 * exported as TS bindings — fine for the V0 internal API.
 */

import { useState } from "react";
import { Link, useFetcher, useLoaderData } from "react-router";
import type { Route } from "./+types/route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { AddButton } from "~/components/ui/AddButton";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Textarea } from "~/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Badge } from "~/components/ui/badge";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { ModelPicker } from "~/components/function/ModelPicker";

export async function loader() {
  const client = getTensorZeroClient();
  const list = await client.listFunctions();
  return { functions: list.functions };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent !== "create") {
    return { ok: false, error: `Unknown intent: ${intent}` };
  }

  const name = String(formData.get("name") ?? "").trim();
  const variant_name =
    String(formData.get("variant_name") ?? "").trim() || "default";
  const model = String(formData.get("model") ?? "").trim();
  const system_template = String(formData.get("system_template") ?? "");

  if (!name || !model) {
    return {
      ok: false,
      error: "Function name and model are required.",
    };
  }

  const variant: Record<string, unknown> = { type: "chat_completion", model };
  if (system_template.trim().length > 0) {
    variant.system_template = {
      __tensorzero_remapped_path: `tensorzero://ui/${name}/${variant_name}/system_template`,
      __data: system_template,
    };
  }
  const config = { type: "chat", variants: { [variant_name]: variant } };

  try {
    const client = getTensorZeroClient();
    const result = await client.createFunction({ name, config });
    return { ok: true, name: result.function.function_name };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

function relativeTime(iso: string): string {
  const ts = new Date(iso);
  if (Number.isNaN(ts.getTime())) return iso;
  const diffMs = Date.now() - ts.getTime();
  const sec = Math.round(diffMs / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  return `${day}d ago`;
}

export default function FunctionsPage() {
  const { functions } = useLoaderData<typeof loader>();
  const fetcher = useFetcher<typeof action>();
  // Pre-show the form when the list is empty — fewer clicks for the
  // first-run "create your first function" flow.
  const [showForm, setShowForm] = useState(functions.length === 0);

  return (
    <PageLayout>
      <PageHeader heading="Functions" count={functions.length} />

      <SectionLayout>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-medium">Active functions</h2>
            <p className="text-fg-muted text-sm">
              Functions you can call via{" "}
              <code className="rounded bg-bg-muted px-1 py-0.5 text-xs">
                POST /inference
              </code>
              .
            </p>
          </div>
          {!showForm && (
            <AddButton label="New function" onAdd={() => setShowForm(true)} />
          )}
        </div>

        {functions.length === 0 ? (
          <EmptyMessage message="No functions yet — use the form below to create one." />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {functions.map((f) => (
                <TableRow key={f.name}>
                  <TableCell>
                    <Link
                      to={`/functions/${encodeURIComponent(f.name)}`}
                      className="font-medium hover:underline"
                    >
                      {f.name}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{f.function_type}</Badge>
                  </TableCell>
                  <TableCell className="text-fg-secondary">
                    {relativeTime(f.created_at)}
                  </TableCell>
                  <TableCell className="text-fg-muted text-sm">
                    {f.creation_source}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </SectionLayout>

      {showForm && (
        <SectionLayout>
          <Card>
            <CardHeader>
              <CardTitle>Create function</CardTitle>
              <CardDescription>
                A function is a named entry point. Every function needs at least
                one variant — you can add more later.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <fetcher.Form method="post" className="flex flex-col gap-4">
                <input type="hidden" name="intent" value="create" />

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="font-medium">Function name</span>
                    <Input
                      name="name"
                      placeholder="my_function"
                      required
                      pattern="[A-Za-z0-9_\-:]+"
                    />
                    <span className="text-fg-muted text-xs">
                      Letters, digits, <code>_-:</code>
                    </span>
                  </label>

                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="font-medium">
                      First variant name{" "}
                      <span className="text-fg-muted font-normal">
                        (optional)
                      </span>
                    </span>
                    <Input
                      name="variant_name"
                      placeholder="default"
                      pattern="[A-Za-z0-9_\-:]+"
                    />
                    <span className="text-fg-muted text-xs">
                      Leave blank to use <code>default</code>.
                    </span>
                  </label>
                </div>

                <div className="flex flex-col gap-1.5 text-sm">
                  <span className="font-medium">Model</span>
                  <ModelPicker name="model" required />
                </div>

                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="font-medium">
                    System template{" "}
                    <span className="text-fg-muted font-normal">
                      (optional)
                    </span>
                  </span>
                  <Textarea
                    name="system_template"
                    placeholder="You are a helpful assistant."
                    rows={4}
                  />
                </label>

                {fetcher.data && !fetcher.data.ok && (
                  <p className="text-destructive text-sm">
                    {fetcher.data.error}
                  </p>
                )}

                <div className="flex items-center gap-2">
                  <Button
                    type="submit"
                    disabled={fetcher.state === "submitting"}
                  >
                    {fetcher.state === "submitting"
                      ? "Creating..."
                      : "Create function"}
                  </Button>
                  {functions.length > 0 && (
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => setShowForm(false)}
                    >
                      Cancel
                    </Button>
                  )}
                </div>
              </fetcher.Form>
            </CardContent>
          </Card>
        </SectionLayout>
      )}
    </PageLayout>
  );
}
