import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Plus, Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { useNavigate } from "react-router";
import {
  toEvaluationUrl,
  toDatapointUrl,
  toDatasetUrl,
  toInferenceUrl,
  toWorkflowEvaluationRunUrl,
} from "~/utils/urls";

interface TagsTableProps {
  tags: Record<string, string>;
  onTagsChange?: (tags: Record<string, string>) => void;
  isEditing: boolean;
}

export function TagsTable({ tags, onTagsChange, isEditing }: TagsTableProps) {
  const navigate = useNavigate();

  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  // Sort tags alphabetically by key
  const sortedTagEntries = Object.entries(tags).sort(([a], [b]) =>
    a.localeCompare(b),
  );

  // Navigation logic from TagsTable component
  // NOTE: tensorzero::dynamic_evaluation_run_id uses the historical tag name.
  // It has been renamed to "workflow_evaluation_run_id" but queries still use the old name.
  // Gateway double-writes both tags. Future migration will update queries to use new tag.
  const navigableKeys = [
    "tensorzero::evaluation_name",
    "tensorzero::dataset_name",
    "tensorzero::evaluator_inference_id",
    "tensorzero::dynamic_evaluation_run_id",
    "tensorzero::workflow_evaluation_run_id",
  ];

  // Add conditional navigation keys
  if (tags["tensorzero::evaluation_name"]) {
    navigableKeys.push("tensorzero::evaluation_run_id");
  }
  if (tags["tensorzero::dataset_name"]) {
    navigableKeys.push("tensorzero::datapoint_id");
  }

  // Navigation handler from TagsTable
  const handleRowClick = (key: string, value: string) => {
    // Only navigate if not in editing mode and navigation is available
    if (!isEditing && navigableKeys.includes(key)) {
      switch (key) {
        case "tensorzero::evaluation_run_id": {
          const evaluationName = tags["tensorzero::evaluation_name"];
          if (!evaluationName) {
            return;
          }
          navigate(
            toEvaluationUrl(evaluationName, { evaluation_run_ids: value }),
          );
          break;
        }
        case "tensorzero::datapoint_id": {
          const datasetName = tags["tensorzero::dataset_name"];
          if (!datasetName) {
            return;
          }
          navigate(toDatapointUrl(datasetName, value));
          break;
        }
        case "tensorzero::evaluation_name":
          navigate(toEvaluationUrl(value));
          break;
        case "tensorzero::dataset_name":
          navigate(toDatasetUrl(value));
          break;
        case "tensorzero::evaluator_inference_id":
          navigate(toInferenceUrl(value));
          break;
        // NOTE: This uses the historical tag name. See comment above navigableKeys definition.
        case "tensorzero::dynamic_evaluation_run_id":
          navigate(toWorkflowEvaluationRunUrl(value));
          break;
        case "tensorzero::workflow_evaluation_run_id":
          navigate(toWorkflowEvaluationRunUrl(value));
          break;
      }
    }
  };

  const handleAddTag = () => {
    const trimmedKey = newKey.trim();
    const trimmedValue = newValue.trim();

    if (trimmedKey && trimmedValue && onTagsChange) {
      // Prevent users from adding system tags
      if (trimmedKey.startsWith("tensorzero::")) {
        return; // Silently ignore system tag attempts
      }

      const updatedTags = { ...tags, [trimmedKey]: trimmedValue };
      onTagsChange(updatedTags);
      setNewKey("");
      setNewValue("");
    }
  };

  const handleRemoveTag = (keyToRemove: string) => {
    if (onTagsChange) {
      const updatedTags = { ...tags };
      delete updatedTags[keyToRemove];
      onTagsChange(updatedTags);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddTag();
    }
  };

  return (
    <div className="space-y-4">
      {/* Tags Table */}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Key</TableHead>
            <TableHead>Value</TableHead>
            {isEditing && <TableHead className="w-12"></TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedTagEntries.length === 0 ? (
            <TableEmptyState message="No tags found" />
          ) : (
            sortedTagEntries.map(([key, value]) => {
              const isSystemTag = key.startsWith("tensorzero::");
              const isNavigable = !isEditing && navigableKeys.includes(key);

              return (
                <TableRow
                  key={key}
                  onClick={() => handleRowClick(key, value)}
                  className={
                    isNavigable ? "hover:bg-bg-subtle cursor-pointer" : ""
                  }
                >
                  <TableCell>
                    <span className="font-mono text-sm">{key}</span>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono text-sm">{value}</span>
                  </TableCell>
                  {isEditing && (
                    <TableCell>
                      {!isSystemTag && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveTag(key)}
                          className="text-muted-foreground hover:text-destructive h-6 w-6 p-0"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      )}
                    </TableCell>
                  )}
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>

      {/* Add new tag form - only show in editing mode */}
      {isEditing && (
        <div className="space-y-3">
          <div className="flex gap-2">
            <Input
              placeholder="Key"
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              onKeyPress={handleKeyPress}
              className={`w-48 font-mono text-sm ${
                newKey.trim().startsWith("tensorzero::")
                  ? "border-destructive focus:border-destructive"
                  : ""
              }`}
            />
            <Input
              placeholder="Value"
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              onKeyPress={handleKeyPress}
              className="w-48 font-mono text-sm"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddTag}
              disabled={
                !newKey.trim() ||
                !newValue.trim() ||
                newKey.trim().startsWith("tensorzero::")
              }
              className="px-3"
            >
              <Plus className="mr-1 h-3 w-3" />
              Add
            </Button>
          </div>
          {newKey.trim().startsWith("tensorzero::") && (
            <p className="text-destructive text-sm">
              System tags (starting with "tensorzero::") cannot be added
              manually.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
