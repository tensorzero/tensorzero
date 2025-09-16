import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { X, Plus, Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { Code } from "~/components/ui/code";

interface TagsEditorProps {
  tags: Record<string, string>;
  onTagsChange: (tags: Record<string, string>) => void;
  isEditing: boolean;
}

export function TagsEditor({ tags, onTagsChange, isEditing }: TagsEditorProps) {
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  // Sort tags alphabetically by key
  const sortedTagEntries = Object.entries(tags).sort(([a], [b]) => a.localeCompare(b));

  const handleAddTag = () => {
    const trimmedKey = newKey.trim();
    const trimmedValue = newValue.trim();
    
    if (trimmedKey && trimmedValue) {
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
    const updatedTags = { ...tags };
    delete updatedTags[keyToRemove];
    onTagsChange(updatedTags);
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
              return (
                <TableRow key={key}>
                  <TableCell>
                    <Code>{key}</Code>
                  </TableCell>
                  <TableCell>
                    <Code>{value}</Code>
                  </TableCell>
                  {isEditing && (
                    <TableCell>
                      {!isSystemTag && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveTag(key)}
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
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
              <Plus className="h-3 w-3 mr-1" />
              Add
            </Button>
          </div>
          {newKey.trim().startsWith("tensorzero::") && (
            <p className="text-sm text-destructive">
              System tags (starting with "tensorzero::") cannot be added manually.
            </p>
          )}
        </div>
      )}
    </div>
  );
}