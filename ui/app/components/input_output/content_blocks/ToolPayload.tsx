import { CodeEditor, useFormattedJson } from "~/components/ui/code-editor";
import { Input } from "~/components/ui/input";

interface ToolPayloadProps {
  id: string;
  name: string;
  payload: string;
  payloadLabel: string;
  isEditing?: boolean;
  onChange?: (id: string, name: string, payload: string) => void;
  enforceJson?: boolean;
}

// `ToolPayload` renders the content of:
// - `ToolCall`: `payload` = `arguments`
// - `ToolResult`: `payload` = `result`
export default function ToolPayload({
  name,
  id,
  payload,
  payloadLabel,
  isEditing,
  onChange,
  enforceJson = false,
}: ToolPayloadProps) {
  const formattedPayload = useFormattedJson(payload);

  return (
    <div className="border-border bg-bg-tertiary/50 grid grid-flow-row grid-cols-[min-content_1fr] grid-rows-[repeat(3,min-content)] place-content-center gap-x-4 gap-y-1 rounded-sm px-3 py-2 text-xs">
      <p className="text-fg-secondary font-medium">Name</p>
      {!isEditing ? (
        <p className="self-center truncate font-mono text-[0.6875rem]">
          {name}
        </p>
      ) : (
        <Input
          type="text"
          value={name}
          data-testid="tool-name-input"
          onChange={(e) => {
            onChange?.(id, e.target.value, payload);
          }}
        />
      )}

      <p className="text-fg-secondary font-medium">ID</p>
      {!isEditing ? (
        <p className="self-center truncate font-mono text-[0.6875rem]">{id}</p>
      ) : (
        <Input
          type="text"
          value={id}
          data-testid="tool-id-input"
          onChange={(e) => {
            onChange?.(e.target.value, name, payload);
          }}
        />
      )}

      <p className="text-fg-secondary font-medium">{payloadLabel}</p>
      <CodeEditor
        allowedLanguages={enforceJson ? ["json"] : undefined}
        value={formattedPayload}
        className="bg-bg-secondary"
        readOnly={!isEditing}
        onChange={(updatedPayload) => {
          onChange?.(id, name, updatedPayload);
        }}
      />
    </div>
  );
}
