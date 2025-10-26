import { type ReactNode, useState } from "react";
import { FileCode } from "lucide-react";
import { z } from "zod";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { Input } from "~/components/ui/input";
import { CodeEditor, useFormattedJson } from "~/components/ui/code-editor";
import { type TemplateInput } from "~/types/tensorzero";
import { JsonValueSchema } from "~/utils/clickhouse/common";

export interface TemplateContentBlockProps {
  block: TemplateInput;
  isEditing?: boolean;
  onChange?: (updatedContentBlock: TemplateInput) => void;
  actionBar?: ReactNode;
}

// Schema for validating `block.arguments`
const templateArgumentsSchema = z.record(
  z.string(),
  JsonValueSchema.optional(),
);

export function TemplateContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: TemplateContentBlockProps) {
  // TODO (GabrielBianconi): there's gotta be a better way to handle this...
  const formattedJson = useFormattedJson(block.arguments);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<FileCode className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        {!isEditing || block.name === "system" ? (
          <div className="flex max-w-full min-w-0 items-center gap-1 overflow-hidden">
            <span className="flex-shrink-0">Template:</span>
            <span
              className="min-w-0 flex-1 truncate font-mono text-xs"
              title={block.name}
            >
              {block.name}
            </span>
          </div>
        ) : (
          <div className="flex max-w-full min-w-0 items-center gap-1">
            <span className="flex-shrink-0">Template:</span>
            <Input
              type="text"
              className="min-w-0 flex-1"
              value={block.name}
              onChange={(e) => {
                onChange?.({
                  ...block,
                  name: e.target.value,
                });
              }}
            />
          </div>
        )}
      </ContentBlockLabel>
      <CodeEditor
        allowedLanguages={["json"]}
        value={formattedJson}
        readOnly={!isEditing}
        // TODO (GabrielBianconi): DANGER! This does not prevent form submission!
        // The user can submit a stale value if an error is present.
        onChange={(updatedArguments) => {
          const validationResult =
            templateArgumentsSchema.safeParse(updatedArguments);

          if (!validationResult.success) {
            setErrorMessage("Invalid JSON Object");
            return;
          }

          setErrorMessage(null);
          onChange?.({
            ...block,
            arguments: validationResult.data,
          });
        }}
      />
      {errorMessage && (
        <div className="text-xs text-red-600">{errorMessage}</div>
      )}
    </div>
  );
}
