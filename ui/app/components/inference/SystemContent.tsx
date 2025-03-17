import React, { useState, useEffect } from "react";
import { Textarea } from "~/components/ui/textarea";

interface SystemContentProps {
  systemContent: string | object;
  isEditing?: boolean;
  onChange?: (value: string | object) => void;
}

export function SystemContent({
  systemContent,
  isEditing = false,
  onChange,
}: SystemContentProps) {
  const [isObject, setIsObject] = useState(typeof systemContent === "object");
  const [displayValue, setDisplayValue] = useState(
    typeof systemContent === "object"
      ? JSON.stringify(systemContent, null, 2)
      : (systemContent as string),
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    // Update display value when systemContent changes externally
    setIsObject(typeof systemContent === "object");
    setDisplayValue(
      typeof systemContent === "object"
        ? JSON.stringify(systemContent, null, 2)
        : (systemContent as string),
    );
  }, [systemContent]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setDisplayValue(newValue);

    if (isObject) {
      try {
        const parsedValue = JSON.parse(newValue);
        setJsonError(null);
        onChange?.(parsedValue);
      } catch {
        setJsonError("Invalid JSON format");
      }
    } else {
      onChange?.(newValue);
    }
  };

  return (
    <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
      <div className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
        System
      </div>
      {isEditing ? (
        <div className="w-full">
          <Textarea
            value={displayValue}
            onChange={handleChange}
            className={`min-h-32 font-mono text-sm ${
              jsonError
                ? "border-red-500 dark:border-red-500"
                : "border-slate-200 dark:border-slate-800"
            }`}
            placeholder="System instructions..."
          />
          {jsonError && (
            <div className="mt-1 text-sm text-red-500">{jsonError}</div>
          )}
        </div>
      ) : (
        <pre className="overflow-x-auto p-4">
          <code className="text-sm">
            {typeof systemContent === "object"
              ? JSON.stringify(systemContent, null, 2)
              : systemContent}
          </code>
        </pre>
      )}
    </div>
  );
}
