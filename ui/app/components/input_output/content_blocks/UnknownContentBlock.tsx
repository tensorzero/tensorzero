import { useState, useEffect } from "react";
import { HelpCircle } from "lucide-react";
import { CodeEditor } from "~/components/ui/code-editor";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import type { JsonValue } from "~/types/tensorzero";

interface UnknownContentBlockProps {
  data: JsonValue;
  isEditing?: boolean;
  onChange?: (data: JsonValue) => void;
  actionBar?: ReactNode;
}

export function UnknownContentBlock({
  data,
  isEditing,
  onChange,
  actionBar,
}: UnknownContentBlockProps) {
  const getDisplayValue = (d: JsonValue) => {
    return JSON.stringify(d, null, 2);
  };

  const isInvalidJson = (value: string) => {
    try {
      JSON.parse(value);
      return false;
    } catch {
      return true;
    }
  };

  const [displayValue, setDisplayValue] = useState(getDisplayValue(data));
  const [jsonError, setJsonError] = useState<string | null>(
    isInvalidJson(displayValue) ? "Invalid JSON format" : null,
  );

  useEffect(() => {
    setDisplayValue(getDisplayValue(data));
    setJsonError(null);
  }, [data]);

  const handleChange = (value: string) => {
    setDisplayValue(value);
    try {
      const parsed = JSON.parse(value);
      setJsonError(null);
      onChange?.(parsed);
    } catch {
      setJsonError("Invalid JSON format");
      // TODO (#4903): Handle invalid intermediate states; right it'll keep stale version (but there is a visual cue)
    }
  };

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<HelpCircle className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Unknown
      </ContentBlockLabel>
      <CodeEditor
        value={displayValue}
        readOnly={!isEditing}
        onChange={isEditing ? handleChange : undefined}
        allowedLanguages={["json"]}
      />
      {isEditing && jsonError && (
        <div className="text-xs text-red-500">{jsonError}</div>
      )}
    </div>
  );
}
