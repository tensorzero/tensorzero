import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { KeyInfo } from "~/types/tensorzero";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { formatDate } from "~/utils/date";
import { Button } from "~/components/ui/button";
import { Pencil, Trash } from "lucide-react";
import { useEffect, useState } from "react";
import { useFetcher } from "react-router";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { Input } from "~/components/ui/input";

function ApiKeyRow({
  apiKey,
  onDelete,
  onEdit,
}: {
  apiKey: KeyInfo;
  onDelete: (publicId: string) => void;
  onEdit: (apiKey: KeyInfo) => void;
}) {
  const isDisabled = apiKey.disabled_at !== undefined;

  const publicIdElement = (
    <code
      className={`inline-block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap ${
        isDisabled ? "cursor-help line-through" : ""
      }`}
    >
      {apiKey.public_id}
    </code>
  );

  return (
    <TableRow
      key={apiKey.public_id}
      id={apiKey.public_id}
      className={isDisabled ? "opacity-50" : ""}
    >
      <TableCell className="w-0">
        {isDisabled ? (
          <Tooltip>
            <TooltipTrigger asChild>{publicIdElement}</TooltipTrigger>
            <TooltipContent>
              Disabled on {formatDate(new Date(apiKey.disabled_at!))}
            </TooltipContent>
          </Tooltip>
        ) : (
          publicIdElement
        )}
      </TableCell>
      <TableCell>
        {apiKey.description ? (
          <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
            {apiKey.description}
          </span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
      </TableCell>
      <TableCell className="w-0">
        <span className="whitespace-nowrap">
          {formatDate(new Date(apiKey.created_at))}
        </span>
      </TableCell>
      <TableCell className="w-0">
        <div className="flex items-center justify-end gap-2">
          <ReadOnlyGuard asChild>
            <Button
              onClick={() => onEdit(apiKey)}
              variant="ghost"
              size="icon"
              className="opacity-60 transition-opacity hover:opacity-100"
            >
              <Pencil className="h-4 w-4" />
            </Button>
          </ReadOnlyGuard>
          <ReadOnlyGuard asChild>
            <Button
              onClick={() => !isDisabled && onDelete(apiKey.public_id)}
              variant="ghost"
              size="icon"
              className="opacity-60 transition-opacity hover:opacity-100"
              disabled={isDisabled}
            >
              <Trash className="h-4 w-4" />
            </Button>
          </ReadOnlyGuard>
        </div>
      </TableCell>
    </TableRow>
  );
}

export default function AuthTable({ apiKeys }: { apiKeys: KeyInfo[] }) {
  const deleteFetcher = useFetcher();
  const updateFetcher = useFetcher();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [keyToEdit, setKeyToEdit] = useState<KeyInfo | null>(null);
  const [editDescription, setEditDescription] = useState("");
  const [shouldCloseAfterSubmit, setShouldCloseAfterSubmit] = useState(false);

  const handleDelete = (publicId: string) => {
    setKeyToDelete(publicId);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = () => {
    if (keyToDelete) {
      deleteFetcher.submit(
        { action: "delete", publicId: keyToDelete },
        { method: "post" },
      );
    }
    setDeleteDialogOpen(false);
    setKeyToDelete(null);
  };

  const handleEdit = (apiKey: KeyInfo) => {
    setKeyToEdit(apiKey);
    setEditDescription(apiKey.description ?? "");
    setShouldCloseAfterSubmit(false);
    setEditDialogOpen(true);
  };

  useEffect(() => {
    if (
      shouldCloseAfterSubmit &&
      editDialogOpen &&
      updateFetcher.state === "idle" &&
      updateFetcher.data?.success
    ) {
      setEditDialogOpen(false);
      setKeyToEdit(null);
      setShouldCloseAfterSubmit(false);
    } else if (
      shouldCloseAfterSubmit &&
      updateFetcher.state === "idle" &&
      updateFetcher.data?.error
    ) {
      setShouldCloseAfterSubmit(false);
    }
  }, [
    shouldCloseAfterSubmit,
    editDialogOpen,
    updateFetcher.state,
    updateFetcher.data,
  ]);

  const handleEditDialogChange = (open: boolean) => {
    setEditDialogOpen(open);
    if (!open) {
      setShouldCloseAfterSubmit(false);
    }
  };

  return (
    <TooltipProvider delayDuration={400}>
      <div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-0 whitespace-nowrap">Public ID</TableHead>
              <TableHead>Description</TableHead>
              <TableHead className="w-0 whitespace-nowrap">Created</TableHead>
              <TableHead className="w-0"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {apiKeys.length === 0 ? (
              <TableEmptyState message="No API keys found" />
            ) : (
              apiKeys.map((apiKey) => (
                <ApiKeyRow
                  key={apiKey.public_id}
                  apiKey={apiKey}
                  onDelete={handleDelete}
                  onEdit={handleEdit}
                />
              ))
            )}
          </TableBody>
        </Table>

        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                Are you sure you want to disable the API key{" "}
                <span className="font-mono text-lg font-bold text-red-500">
                  {keyToDelete}
                </span>
                ?
              </DialogTitle>
              <DialogDescription>
                This will disable the API key and it will no longer be able to
                authenticate. This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="flex justify-between gap-2">
              <Button
                variant="outline"
                onClick={() => setDeleteDialogOpen(false)}
              >
                Cancel
              </Button>
              <div className="flex-1" />
              <Button variant="destructive" onClick={confirmDelete}>
                <Trash className="inline h-4 w-4" />
                Disable
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={editDialogOpen} onOpenChange={handleEditDialogChange}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit API key description</DialogTitle>
              <DialogDescription>
                Update the description for API key{" "}
                <span className="font-mono font-semibold">
                  {keyToEdit?.public_id}
                </span>
                .
              </DialogDescription>
            </DialogHeader>
            <updateFetcher.Form
              method="post"
              className="space-y-4"
              onSubmit={() => setShouldCloseAfterSubmit(true)}
            >
              <input type="hidden" name="action" value="update" />
              <input
                type="hidden"
                name="publicId"
                value={keyToEdit?.public_id ?? ""}
              />
              <Input
                name="description"
                value={editDescription}
                onChange={(event) => setEditDescription(event.target.value)}
                placeholder="Optional description"
                autoFocus
              />
              {updateFetcher.data?.error ? (
                <p className="text-sm text-red-500">
                  {updateFetcher.data.error}
                </p>
              ) : null}
              <DialogFooter className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEditDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={updateFetcher.state !== "idle"}>
                  Save
                </Button>
              </DialogFooter>
            </updateFetcher.Form>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
}
