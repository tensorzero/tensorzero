import { AlignLeftIcon, EyeIcon, CodeIcon } from "lucide-react";
import { useFormattedJson } from "~/components/ui/code-editor";
import { VirtualizedCodeEditor } from "~/components/ui/virtualized-code-editor";
import { type ReactNode, useMemo, useState } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { Markdown } from "~/components/ui/markdown";

/** Simple check for markdown-like content */
function looksLikeMarkdown(text: string): boolean {
  return (
    text.includes("# ") ||
    text.includes("## ") ||
    text.includes("**") ||
    text.includes("__") ||
    (text.includes("[") && text.includes("](")) ||
    text.includes("```") ||
    /^\d+\.\s/m.test(text) ||
    /^[-*]\s/m.test(text)
  );
}

interface TextContentBlockProps {
  label: string;
  text: string;
  isEditing?: boolean;
  onChange?: (value: string) => void;
  actionBar?: ReactNode;
}

export function TextContentBlock({
  label,
  text,
  isEditing,
  onChange,
  actionBar,
}: TextContentBlockProps) {
  const formattedText = useFormattedJson(text);
  const isMarkdown = useMemo(() => looksLikeMarkdown(text), [text]);
  const [showRendered, setShowRendered] = useState(false);

  const renderToggle =
    isMarkdown && !isEditing ? (
      <button
        type="button"
        className="text-fg-muted hover:text-fg-primary ml-auto flex cursor-pointer items-center gap-1 text-xs transition-colors"
        onClick={() => setShowRendered((v) => !v)}
        title={showRendered ? "Show source" : "Show rendered"}
      >
        {showRendered ? (
          <CodeIcon className="h-3 w-3" />
        ) : (
          <EyeIcon className="h-3 w-3" />
        )}
        {showRendered ? "Source" : "Preview"}
      </button>
    ) : null;

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<AlignLeftIcon className="text-fg-muted h-3 w-3" />}
        actionBar={
          <>
            {actionBar}
            {renderToggle}
          </>
        }
      >
        {label}
      </ContentBlockLabel>
      {showRendered && !isEditing ? (
        <div className="bg-bg-primary border-border max-h-[400px] overflow-auto rounded-md border px-4 py-3">
          <Markdown>{text}</Markdown>
        </div>
      ) : (
        <VirtualizedCodeEditor
          value={formattedText}
          readOnly={!isEditing}
          onChange={onChange}
        />
      )}
    </div>
  );
}
