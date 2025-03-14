import React from "react";
import { Textarea } from "~/components/ui/textarea";

interface SystemContentProps {
  systemContent: string | object;
  isEditing?: boolean;
  onChange?: (value: string) => void;
}

export function SystemContent({
  systemContent,
  isEditing = false,
  onChange,
}: SystemContentProps) {
  return (
    <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
      <div className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
        System
      </div>
      {isEditing ? (
        <Textarea
          value={
            typeof systemContent === "object"
              ? JSON.stringify(systemContent, null, 2)
              : (systemContent as string)
          }
          onChange={(e) => onChange?.(e.target.value)}
          className="min-h-32 font-mono text-sm"
          placeholder="System instructions..."
        />
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
