import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { X, Plus } from "lucide-react";

interface TagsEditorProps {
  tags: Record<string, string>;
  onTagsChange: (tags: Record<string, string>) => void;
  isEditing: boolean;
}

export function TagsEditor({ tags, onTagsChange, isEditing }: TagsEditorProps) {
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  if (!isEditing) {
    // Display mode - show tags as badges
    const tagEntries = Object.entries(tags);
    if (tagEntries.length === 0) {
      return <span className="text-muted-foreground text-sm">—</span>;
    }
    
    return (
      <div className="flex flex-wrap gap-1">
        {tagEntries.map(([key, value]) => {
          const isSystemTag = key.startsWith("tensorzero::");
          return (
            <Badge
              key={key}
              variant={isSystemTag ? "secondary" : "outline"}
              className="font-mono text-xs"
            >
              {key}={value}
            </Badge>
          );
        })}
      </div>
    );
  }

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
    <div className="space-y-3">
      {/* Existing tags */}
      {Object.entries(tags).length > 0 && (
        <div className="space-y-2">
          {Object.entries(tags).map(([key, value]) => {
            const isSystemTag = key.startsWith("tensorzero::");
            return (
              <div key={key} className="flex items-center gap-2">
                <Badge
                  variant={isSystemTag ? "secondary" : "outline"}
                  className="font-mono text-xs"
                >
                  {key}={value}
                </Badge>
                {!isSystemTag && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveTag(key)}
                    className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Add new tag form */}
      <div className="flex gap-2">
        <Input
          placeholder="Key"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyPress={handleKeyPress}
          className={`w-32 font-mono text-sm ${
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
          className="w-32 font-mono text-sm"
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
          <Plus className="h-3 w-3" />
        </Button>
      </div>
      {newKey.trim().startsWith("tensorzero::") && (
        <p className="text-sm text-destructive">
          System tags (starting with "tensorzero::") cannot be added manually.
        </p>
      )}
    </div>
  );
}