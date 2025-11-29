import { useState, useEffect } from "react";
import { HelpCircle } from "lucide-react";
import { CodeEditor } from "~/components/ui/code-editor";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import type { JsonValue } from "~/types/tensorzero";

// Marker type for invalid JSON - allows validation at save time
export interface InvalidJsonMarker {
  __invalid_json__: true;
  raw: string;
}

export function isInvalidJsonMarker(
  value: unknown,
): value is InvalidJsonMarker {
  return (
    typeof value === "object" &&
    value !== null &&
    "__invalid_json__" in value &&
    (value as InvalidJsonMarker).__invalid_json__ === true
  );
}

interface UnknownContentBlockProps {
  data: JsonValue | InvalidJsonMarker;
  isEditing?: boolean;
  onChange?: (data: JsonValue | InvalidJsonMarker) => void;
  actionBar?: ReactNode;
}

export function UnknownContentBlock({
  data,
  isEditing,
  onChange,
  actionBar,
}: UnknownContentBlockProps) {
  const getDisplayValue = (d: JsonValue | InvalidJsonMarker) => {
    if (isInvalidJsonMarker(d)) {
      return d.raw;
    }
    return JSON.stringify(d, null, 2);
  };

  const [displayValue, setDisplayValue] = useState(getDisplayValue(data));
  const [jsonError, setJsonError] = useState<string | null>(
    isInvalidJsonMarker(data) ? "Invalid JSON format" : null,
  );

  useEffect(() => {
    setDisplayValue(getDisplayValue(data));
    setJsonError(isInvalidJsonMarker(data) ? "Invalid JSON format" : null);
  }, [data]);

  const handleChange = (value: string) => {
    setDisplayValue(value);
    try {
      const parsed = JSON.parse(value);
      setJsonError(null);
      onChange?.(parsed);
    } catch {
      setJsonError("Invalid JSON format");
      // Store the invalid JSON with a marker so validation can detect it
      onChange?.({ __invalid_json__: true, raw: value });
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
