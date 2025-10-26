import { AlignLeftIcon } from "lucide-react";
import { CodeEditor, useFormattedJson } from "~/components/ui/code-editor";
import { type ReactNode } from "react";
import ContentBlockLabel from "~/components/input_output/content_blocks/ContentBlockLabel";

interface TextContentBlockProps {
  label: string;
  text: string;
  isEditing?: boolean;
  onChange?: (value: string) => void;
  actionBar?: ReactNode;
}

export default function TextContentBlock({
  label,
  text: content,
  // footer,
  isEditing,
  onChange,
  actionBar,
}: TextContentBlockProps) {
  // TODO (GabrielBianconi): there's gotta be a better way to handle this...
  const formattedContent = useFormattedJson(content);

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<AlignLeftIcon className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        {label}
      </ContentBlockLabel>
      <CodeEditor
        value={formattedContent}
        readOnly={!isEditing}
        onChange={onChange}
      />
      {/*{footer ? (
        <div className="text-fg-tertiary text-xs font-medium">{footer}</div>
      ) : null}*/}
    </div>
  );
}
