/**
 * `/functions/:name` — view a function and its variants. Adds a variant via
 * `POST /internal/functions/:name/variants`.
 *
 * Variant editing happens by reading the live shape, replacing one variant,
 * and PATCHing the function back. The `expected_current_function_version_id`
 * CAS check guards against concurrent edits — the button label "Update"
 * implies a +1 to the variant's version field, so we let the server compute
 * the exact integer (this UI just submits it as-is for now).
 */

import { useState } from "react";
import { Link, useFetcher, useLoaderData } from "react-router";
import type { Route } from "./+types/route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { AddButton } from "~/components/ui/AddButton";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Textarea } from "~/components/ui/textarea";
import { Badge } from "~/components/ui/badge";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { ModelPicker } from "~/components/function/ModelPicker";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemContent,
  BasicInfoItemTitle,
} from "~/components/layout/BasicInfoLayout";
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

export async function loader({ params }: Route.LoaderArgs) {
  const name = params.name!;
  const client = getTensorZeroClient();
  const [fn, variants] = await Promise.all([
    client.getFunction(name),
    client.listVariants(name),
  ]);
  return { name, function: fn, variants: variants.variants };
}

export async function action({ request, params }: Route.ActionArgs) {
  const name = params.name!;
  const formData = await request.formData();
  const intent = formData.get("intent");
  const expected_version = String(
    formData.get("expected_current_function_version_id") ?? "",
  );
  const client = getTensorZeroClient();

  try {
    if (intent === "add_variant") {
      const variant_name = String(formData.get("variant_name") ?? "").trim();
      const model = String(formData.get("model") ?? "").trim();
      const system_template = String(formData.get("system_template") ?? "");

      if (!variant_name || !model) {
        return { ok: false, error: "Variant name and model are required." };
      }

      // No `version` in the create payload: a brand-new variant starts
      // unversioned (effectively v0). Subsequent edits via PATCH bump
      // the version automatically server-side, so the UI never needs to
      // compute or display a "next version".
      const variantConfig: Record<string, unknown> = {
        type: "chat_completion",
        model,
      };
      if (system_template.trim().length > 0) {
        variantConfig.system_template = {
          __tensorzero_remapped_path: `tensorzero://ui/${name}/${variant_name}/system_template`,
          __data: system_template,
        };
      }

      await client.createVariant(name, {
        variant_name,
        config: variantConfig,
        expected_current_function_version_id: expected_version,
      });
      return { ok: true };
    }

    if (intent === "delete_function") {
      await client.deleteFunction(name, expected_version);
      return { ok: true, deleted: true };
    }

    return { ok: false, error: `Unknown intent: ${intent}` };
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

export default function FunctionDetailPage() {
  const { name, function: fn, variants } = useLoaderData<typeof loader>();
  const fetcher = useFetcher<typeof action>();
  // Pre-show the form on first visit (no variants yet) so the user lands
  // straight on the input fields they came here to fill.
  const [showAddVariant, setShowAddVariant] = useState(variants.length === 0);

  return (
    <PageLayout>
      <PageHeader
        eyebrow={
          <Link to="/functions" className="hover:underline">
            ← Functions
          </Link>
        }
        name={name}
        count={variants.length}
      />

      <SectionLayout>
        <SectionHeader heading="Details" />
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Created</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <span title={fn.created_at}>{relativeTime(fn.created_at)}</span>
            </BasicInfoItemContent>
          </BasicInfoItem>
          <BasicInfoItem>
            <BasicInfoItemTitle>Source</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Badge variant="secondary">{fn.creation_source}</Badge>
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </SectionLayout>

      <SectionLayout>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-medium">Variants</h2>
            <p className="text-fg-muted text-sm">
              Each variant is one routable implementation of this function.
            </p>
          </div>
          {!showAddVariant && (
            <AddButton
              label="Add variant"
              onAdd={() => setShowAddVariant(true)}
            />
          )}
        </div>

        {variants.length === 0 ? (
          <EmptyMessage message="No variants yet — add one before this function can serve inferences." />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {variants.map((v) => (
                <TableRow key={v.name}>
                  <TableCell className="font-medium">{v.name}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{v.variant_type}</Badge>
                  </TableCell>
                  <TableCell className="text-fg-secondary">
                    {v.version === 0 ? (
                      <span className="text-fg-muted">—</span>
                    ) : (
                      <code className="text-xs">v{v.version}</code>
                    )}
                  </TableCell>
                  <TableCell className="text-fg-secondary" title={v.created_at}>
                    {relativeTime(v.created_at)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </SectionLayout>

      {showAddVariant && (
        <SectionLayout>
          <Card>
            <CardHeader>
              <CardTitle>Add variant</CardTitle>
              <CardDescription>
                A new variant lives alongside any existing ones. Traffic splits
                are configured separately via{" "}
                <code className="text-xs">experimentation</code>.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <fetcher.Form method="post" className="flex flex-col gap-4">
                <input type="hidden" name="intent" value="add_variant" />
                <input
                  type="hidden"
                  name="expected_current_function_version_id"
                  value={fn.function_version_id}
                />

                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="font-medium">Variant name</span>
                  <Input
                    name="variant_name"
                    placeholder="my_variant"
                    required
                    pattern="[A-Za-z0-9_\-:]+"
                  />
                  <span className="text-fg-muted text-xs">
                    Versions are managed automatically — the first save is
                    unversioned, and each subsequent edit (PATCH) increments the
                    version by one.
                  </span>
                </label>

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
                      ? "Adding..."
                      : "Add variant"}
                  </Button>
                  {variants.length > 0 && (
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => setShowAddVariant(false)}
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

      <SectionLayout>
        <Link to="/playground" className="text-primary text-sm hover:underline">
          Try this function in the Playground →
        </Link>
      </SectionLayout>
    </PageLayout>
  );
}
