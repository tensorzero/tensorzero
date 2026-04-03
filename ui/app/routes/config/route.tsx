import { data, useFetcher, useRevalidator } from "react-router";
import {
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  AlertTriangle,
  Loader2,
  PencilLine,
  Plus,
  RefreshCcw,
  Save,
  Trash2,
} from "lucide-react";
import { z } from "zod";

import type { Route } from "./+types/route";
import { CodeEditor } from "~/components/ui/code-editor";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Input } from "~/components/ui/input";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { useReadOnly } from "~/context/read-only";
import { useToast } from "~/hooks/use-toast";
import { cn } from "~/utils/common";
import { invalidateConfigCache } from "~/utils/config/index.server";
import { loadFeatureFlags } from "~/utils/feature_flags.server";
import { logger } from "~/utils/logger";
import { isReadOnlyMode } from "~/utils/read-only.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { TensorZeroServerError } from "~/utils/tensorzero";
import type {
  ApplyConfigTomlRequest,
  ApplyConfigTomlResponse,
  GetConfigTomlResponse,
} from "~/types/tensorzero";
import { createNewConfigPath } from "./editor-utils";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { HelpTooltip } from "~/components/ui/HelpTooltip";

const ConfigEditorSubmissionSchema = z.object({
  toml: z.string(),
  path_contents: z.record(z.string()),
  base_signature: z.string(),
});

type ConfigEditorSubmission = z.infer<typeof ConfigEditorSubmissionSchema>;

/**
 * Unified editable-config shape used by the route state. Both
 * `GetConfigTomlResponse` and `ApplyConfigTomlResponse` satisfy this — we only
 * care about the fields needed by the editor UI.
 */
type EditableConfig = Pick<
  GetConfigTomlResponse,
  "toml" | "path_contents" | "hash" | "base_signature"
>;

type ActionData =
  | { success: true; editableConfig: ApplyConfigTomlResponse }
  | {
      success: false;
      error: string;
      conflict: boolean;
      latestConfig?: GetConfigTomlResponse;
    };

export async function loader(_args: Route.LoaderArgs) {
  const featureFlags = loadFeatureFlags();
  if (!featureFlags.configEditor) {
    throw data("Not found", { status: 404 });
  }

  const editableConfig = await getTensorZeroClient().getConfigToml();
  return { editableConfig };
}

export async function action({ request }: Route.ActionArgs) {
  const featureFlags = loadFeatureFlags();
  if (!featureFlags.configEditor) {
    throw data("Not found", { status: 404 });
  }
  if (isReadOnlyMode()) {
    return data<ActionData>(
      {
        success: false,
        conflict: false,
        error: "Config editing is not available in read-only mode.",
      },
      { status: 403 },
    );
  }

  const submission = ConfigEditorSubmissionSchema.parse(await request.json());

  try {
    const editableConfig = await getTensorZeroClient().applyConfigToml(
      submission satisfies ApplyConfigTomlRequest,
    );
    invalidateConfigCache();
    return data<ActionData>({ success: true, editableConfig });
  } catch (error) {
    if (error instanceof TensorZeroServerError && error.status === 409) {
      const latestConfig = await getTensorZeroClient().getConfigToml();
      invalidateConfigCache();
      return data<ActionData>(
        {
          success: false,
          conflict: true,
          error: error.message,
          latestConfig,
        },
        { status: 409 },
      );
    }

    logger.error("Failed to apply config TOML", error);
    return data<ActionData>(
      {
        success: false,
        conflict: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to apply config changes.",
      },
      { status: 500 },
    );
  }
}

export default function ConfigEditorRoute({
  loaderData,
}: Route.ComponentProps) {
  const fetcher = useFetcher<ActionData>();
  const revalidator = useRevalidator();
  const { toast } = useToast();
  const isReadOnly = useReadOnly();
  const hasShownToastRef = useRef(false);

  const [toml, setToml] = useState(loaderData.editableConfig.toml);
  const [pathContents, setPathContents] = useState(
    loaderData.editableConfig.path_contents,
  );
  const [baseSignature, setBaseSignature] = useState(
    loaderData.editableConfig.base_signature,
  );
  const [_hash, setHash] = useState(loaderData.editableConfig.hash);
  const [savedConfig, setSavedConfig] = useState<EditableConfig>(
    loaderData.editableConfig,
  );
  const [selectedPath, setSelectedPath] = useState<string | null>(
    getInitialSelectedPath(loaderData.editableConfig.path_contents),
  );
  const [renameDraft, setRenameDraft] = useState(selectedPath ?? "");
  const [conflictConfig, setConflictConfig] = useState<EditableConfig | null>(
    null,
  );
  const [inlineError, setInlineError] = useState<string | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(220);

  const filePaths = useMemo(
    () =>
      Object.keys(pathContents).sort((left, right) =>
        left.localeCompare(right),
      ),
    [pathContents],
  );

  const selectedFileContents = selectedPath
    ? (pathContents[selectedPath] ?? "")
    : "";
  const isSaving = fetcher.state !== "idle";
  const isDirty =
    toml !== savedConfig.toml ||
    JSON.stringify(pathContents) !==
      JSON.stringify(savedConfig.path_contents) ||
    baseSignature !== savedConfig.base_signature;

  useEffect(() => {
    setRenameDraft(selectedPath ?? "");
  }, [selectedPath]);

  useEffect(() => {
    if (!selectedPath && filePaths.length > 0) {
      setSelectedPath(filePaths[0]);
      return;
    }
    if (selectedPath && !(selectedPath in pathContents)) {
      setSelectedPath(filePaths[0] ?? null);
    }
  }, [filePaths, pathContents, selectedPath]);

  useEffect(() => {
    if (isDirty) {
      return;
    }
    if (
      savedConfig.base_signature === loaderData.editableConfig.base_signature
    ) {
      return;
    }
    setSavedConfig(loaderData.editableConfig);
    applyEditableConfig(loaderData.editableConfig, {
      setToml,
      setPathContents,
      setBaseSignature,
      setHash,
      setSelectedPath,
    });
  }, [isDirty, loaderData.editableConfig, savedConfig.base_signature]);

  useEffect(() => {
    const actionData = fetcher.data;
    if (fetcher.state === "submitting") {
      hasShownToastRef.current = false;
      setInlineError(null);
      return;
    }

    if (fetcher.state !== "idle" || !actionData || hasShownToastRef.current) {
      return;
    }

    hasShownToastRef.current = true;

    if (actionData.success) {
      startTransition(() => {
        applyEditableConfig(actionData.editableConfig, {
          setToml,
          setPathContents,
          setBaseSignature,
          setHash,
          setSelectedPath,
        });
        setSavedConfig(actionData.editableConfig);
        setConflictConfig(null);
        setInlineError(null);
      });
      toast.success({
        title: "Config applied",
        description:
          "If you changed clickhouse, postgres, or rate limiting configs, restart the gateway for them to take effect.",
      });
      revalidator.revalidate();
      return;
    }

    const errorData = actionData as Extract<ActionData, { success: false }>;
    setInlineError(errorData.error);
    if (errorData.conflict) {
      setConflictConfig(errorData.latestConfig ?? null);
      toast.error({
        title: "Conflict when applying config",
        description: "Reload the latest snapshot before applying your edits.",
      });
      return;
    }

    toast.error({
      title: "Failed to apply config",
    });
  }, [fetcher.state, fetcher.data, revalidator, toast]);

  const handleSave = () => {
    setInlineError(null);
    setConflictConfig(null);
    fetcher.submit(
      {
        toml,
        path_contents: pathContents,
        base_signature: baseSignature,
      } satisfies ConfigEditorSubmission,
      {
        method: "POST",
        encType: "application/json",
      },
    );
  };

  const handlePathRename = () => {
    if (!selectedPath) {
      return;
    }

    const trimmed = renameDraft.trim();
    if (!trimmed) {
      setInlineError("Path name cannot be empty.");
      return;
    }
    if (trimmed !== selectedPath && trimmed in pathContents) {
      setInlineError(`Path \`${trimmed}\` already exists.`);
      return;
    }

    setPathContents((current) => {
      const next: Record<string, string> = {};
      for (const [path, contents] of Object.entries(current)) {
        next[path === selectedPath ? trimmed : path] = contents;
      }
      return next;
    });
    setSelectedPath(trimmed);
    setInlineError(null);
  };

  const handleAddPath = () => {
    const newPath = createNewConfigPath(filePaths);
    setPathContents((current) => ({ ...current, [newPath]: "" }));
    setSelectedPath(newPath);
    setRenameDraft(newPath);
    setInlineError(null);
  };

  const handleDeletePath = () => {
    if (!selectedPath) {
      return;
    }
    setPathContents((current) => {
      const next = { ...current };
      delete next[selectedPath];
      return next;
    });
    setInlineError(null);
  };

  const handleDividerMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = sidebarWidth;
    const onMouseMove = (ev: MouseEvent) => {
      setSidebarWidth(
        Math.max(140, Math.min(480, startWidth + ev.clientX - startX)),
      );
    };
    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  };

  const handleReloadLatest = () => {
    if (!conflictConfig) {
      return;
    }
    startTransition(() => {
      applyEditableConfig(conflictConfig, {
        setToml,
        setPathContents,
        setBaseSignature,
        setHash,
        setSelectedPath,
      });
      setSavedConfig(conflictConfig);
      setConflictConfig(null);
      setInlineError(null);
    });
    revalidator.revalidate();
  };

  return (
    <PageLayout>
      <PageHeader
        heading="Configuration"
        tag={isDirty && <Badge variant="secondary">Unsaved edits</Badge>}
      />

      <SectionsGroup>
        <SectionLayout>
          <div className="flex items-center justify-between gap-4">
            <SectionHeader heading="Config TOML" />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (conflictConfig) {
                    handleReloadLatest();
                  } else {
                    setConflictConfig(null);
                    setInlineError(null);
                    revalidator.revalidate();
                  }
                }}
                disabled={isSaving || (isDirty && !conflictConfig)}
              >
                <RefreshCcw className="mr-2 h-4 w-4" />
                Reload latest
              </Button>
              <ReadOnlyGuard asChild>
                <Button
                  size="sm"
                  onClick={handleSave}
                  disabled={isReadOnly || !isDirty || isSaving}
                >
                  {isSaving ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Save className="mr-2 h-4 w-4" />
                  )}
                  Save config and files
                </Button>
              </ReadOnlyGuard>
            </div>
          </div>

          {(inlineError || conflictConfig) && (
            <div className="border-border bg-bg-secondary flex flex-col gap-3 rounded-lg border px-4 py-3">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <div className="space-y-1">
                  <p className="font-medium">
                    {conflictConfig ? "Apply conflict" : "Apply failed"}
                  </p>
                  {inlineError && (
                    <p className="text-muted-foreground text-sm">
                      {inlineError}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          <div
            className="border-border resize-y overflow-hidden rounded-md border"
            style={{ height: "55vh", minHeight: "8rem" }}
          >
            <CodeEditor
              ariaLabel="Editable config TOML"
              // TODO: add TOML support
              allowedLanguages={["text"]}
              value={toml}
              onChange={setToml}
              readOnly={isReadOnly}
              height="100%"
            />
          </div>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader
            heading="Referenced Files"
            help={
              <HelpTooltip>
                Edit prompt templates, JSON schemas, and other files referenced
                in the config TOML. File name are unique strings, and must be
                referenced exactly in the TOML.
              </HelpTooltip>
            }
          />

          <div
            className="resize-y overflow-hidden"
            style={{ height: "55vh", minHeight: "10rem" }}
          >
            <div
              className="grid h-full"
              style={{
                gridTemplateColumns: `${sidebarWidth}px 1rem minmax(0, 1fr)`,
              }}
            >
              <div className="bg-bg-secondary flex h-full flex-col overflow-hidden rounded-md border">
                <div className="border-border flex items-center justify-between px-3 py-2">
                  <p className="text-muted-foreground text-xs font-medium uppercase tracking-[0.14em]">
                    Files
                  </p>
                  <ReadOnlyGuard asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={handleAddPath}
                      disabled={isReadOnly}
                      aria-label="New file"
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </ReadOnlyGuard>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto p-2">
                  {filePaths.length === 0 ? (
                    <p className="text-muted-foreground px-2 py-3 text-sm">
                      No path-backed content in this snapshot.
                    </p>
                  ) : (
                    <div className="flex flex-col gap-1">
                      {filePaths.map((path) => (
                        <button
                          key={path}
                          type="button"
                          className={cn(
                            "hover:bg-accent text-left rounded-md px-2 py-2 text-sm transition-colors",
                            path === selectedPath &&
                              "bg-accent text-accent-foreground",
                          )}
                          onClick={() => setSelectedPath(path)}
                        >
                          <div className="truncate font-medium">{path}</div>
                          <div className="text-muted-foreground mt-0.5 text-xs">
                            {pathContents[path]?.length ?? 0} chars
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Drag handle — occupies the 1rem gap column */}
              <div
                className="group flex cursor-col-resize items-stretch justify-center"
                onMouseDown={handleDividerMouseDown}
              >
                <div className="w-px rounded-full bg-transparent transition-colors group-hover:bg-border" />
              </div>

              <div className="flex h-full flex-col gap-4 overflow-hidden">
                {selectedPath ? (
                  <>
                    <div className="flex-none space-y-3">
                      <div className="flex gap-2">
                        <Input
                          value={renameDraft}
                          onChange={(event) =>
                            setRenameDraft(event.target.value)
                          }
                          aria-label="File name"
                          placeholder="File name"
                          disabled={isReadOnly}
                        />
                        <ReadOnlyGuard asChild>
                          <Button
                            variant="outline"
                            onClick={handlePathRename}
                            disabled={isReadOnly}
                          >
                            <PencilLine />
                            Rename
                          </Button>
                        </ReadOnlyGuard>
                        <ReadOnlyGuard asChild>
                          <Button
                            variant="outline"
                            onClick={handleDeletePath}
                            disabled={isReadOnly}
                          >
                            <Trash2 />
                            Delete
                          </Button>
                        </ReadOnlyGuard>
                      </div>
                    </div>
                    <div className="border-border flex-1 overflow-hidden rounded-md border">
                      <CodeEditor
                        ariaLabel={`Editable content for ${selectedPath}`}
                        value={selectedFileContents}
                        onChange={(value) => {
                          setPathContents((current) => ({
                            ...current,
                            [selectedPath]: value,
                          }));
                        }}
                        readOnly={isReadOnly}
                        height="100%"
                      />
                    </div>
                  </>
                ) : (
                  <div className="text-muted-foreground flex flex-1 items-center justify-center rounded-md border border-dashed text-sm">
                    Select a path to edit its contents.
                  </div>
                )}
              </div>
            </div>
          </div>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

function getInitialSelectedPath(
  pathContents: Record<string, string>,
): string | null {
  const [firstPath] = Object.keys(pathContents).sort((left, right) =>
    left.localeCompare(right),
  );
  return firstPath ?? null;
}

function applyEditableConfig(
  editableConfig: EditableConfig,
  setters: {
    setToml: Dispatch<SetStateAction<string>>;
    setPathContents: Dispatch<SetStateAction<Record<string, string>>>;
    setBaseSignature: Dispatch<SetStateAction<string>>;
    setHash: Dispatch<SetStateAction<string>>;
    setSelectedPath: Dispatch<SetStateAction<string | null>>;
  },
) {
  setters.setToml(editableConfig.toml);
  setters.setPathContents(editableConfig.path_contents);
  setters.setBaseSignature(editableConfig.base_signature);
  setters.setHash(editableConfig.hash);
  setters.setSelectedPath(getInitialSelectedPath(editableConfig.path_contents));
}
