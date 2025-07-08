import { useMemo, useState } from "react";
import { useCopy } from "~/hooks/use-copy";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { StreamLanguage } from "@codemirror/language";
import { jinja2 } from "@codemirror/legacy-modes/mode/jinja2";
import { githubLightInit } from "@uiw/codemirror-theme-github";
import { EditorView } from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import { Button } from "./button";
import { cn } from "~/utils/common";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./dropdown-menu";
import {
  ChevronDownIcon,
  WrapTextIcon,
  CheckCheckIcon,
  ClipboardIcon,
} from "lucide-react";

export type Language = "json" | "markdown" | "jinja2" | "text";

export interface CodeEditorProps {
  value?: string;
  onChange?: (value: string) => void;
  onBlur?: () => void;
  /** Languages to show in selector. If auto detect is disabled, defaults to the first one. */
  allowedLanguages?: [Language, ...Language[]];
  autoDetectLanguage?: boolean;
  readOnly?: boolean;
  showLineNumbers?: boolean;
  placeholder?: string;
  className?: string;
}

const LANGUAGE_EXTENSIONS = {
  json: [json()],
  markdown: [markdown()],
  jinja2: [StreamLanguage.define(jinja2)],
  text: [],
} as const;

const LANGUAGE_LABELS = {
  json: "JSON",
  markdown: "Markdown",
  jinja2: "Template",
  text: "Text",
} as const satisfies {
  [language: string]: string;
};

function detectLanguage(content: string): Language {
  const trimmed = content.trim();

  if (!trimmed) {
    return "text";
  }

  // JSON detection
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      JSON.parse(trimmed);
      return "json";
    } catch {
      // Not valid JSON, continue checking
    }
  }

  // Jinja2 detection
  if (
    content.includes("{{") ||
    content.includes("{%") ||
    content.includes("{#")
  ) {
    return "jinja2";
  }

  // Markdown detection (basic heuristics)
  if (
    content.includes("# ") ||
    content.includes("## ") ||
    content.includes("**") ||
    content.includes("__") ||
    (content.includes("[") && content.includes("](")) ||
    content.includes("```")
  ) {
    return "markdown";
  }

  return "text";
}

const DEFAULT_WORD_WRAP_LANGUAGES: Language[] = ["text", "jinja2", "markdown"];

export const CodeEditor: React.FC<CodeEditorProps> = ({
  value = "",
  onChange,
  onBlur,
  allowedLanguages = ["text", "markdown", "json", "jinja2"],
  autoDetectLanguage = allowedLanguages.length > 1,
  readOnly = false,
  showLineNumbers = true,
  placeholder,
  className,
}) => {
  const [language, setLanguage] = useState<Language>(() =>
    autoDetectLanguage ? detectLanguage(value) : allowedLanguages[0],
  );

  const [wordWrap, setWordWrap] = useState(
    DEFAULT_WORD_WRAP_LANGUAGES.includes(language),
  );
  const { copy, didCopy, isCopyAvailable } = useCopy();

  // Custom theme to remove dotted border and add focus styles
  const extensions = useMemo(() => {
    const customTheme = EditorView.theme({
      ".cm-focused": {
        outline: "none !important",
      },
      ".cm-editor": {
        borderRadius: "0.375rem", // rounded-md
      },
    });

    const exts: Extension[] = [...LANGUAGE_EXTENSIONS[language], customTheme];

    // Add line wrapping extension
    if (wordWrap) {
      exts.push(EditorView.lineWrapping);
    }

    // Add read-only extension
    if (readOnly) {
      exts.push(EditorView.editable.of(false));
    }

    return exts;
  }, [language, wordWrap, readOnly]);

  return (
    <div className={cn("group relative", className)}>
      {/* TODO: for focus-within to work properly, `DropdownMenu` cannot render with a portal or the buttons will disappear when the dropdown is open - best way to resolve? */}
      <div className="absolute top-1 right-1 z-10 flex gap-1.5 opacity-0 transition-opacity duration-200 group-hover:opacity-100 focus-within:opacity-100">
        {isCopyAvailable && (
          <Button
            variant="ghost"
            size="iconSm"
            onClick={() => copy(value)}
            className="h-6 w-6 p-3 text-xs"
            title={didCopy ? "Copied!" : "Copy to clipboard"}
          >
            {didCopy ? (
              <CheckCheckIcon className="h-2 w-2" />
            ) : (
              <ClipboardIcon className="h-2 w-2" />
            )}
          </Button>
        )}

        {/* TODO Style a custom variant for these buttons - they don't match sizing/style of dropdown */}
        <Button
          variant={wordWrap ? "default" : "ghost"}
          size="iconSm"
          onClick={() => setWordWrap((wrap) => !wrap)}
          aria-pressed={wordWrap}
          className="h-6 w-6 p-3 text-xs"
          title="Toggle word wrap"
        >
          <WrapTextIcon className="h-2 w-2" />
        </Button>

        {allowedLanguages.length > 1 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-6 gap-1 px-2 py-1 text-xs"
              >
                {LANGUAGE_LABELS[language]}
                <ChevronDownIcon className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {allowedLanguages.map((languageOption) => (
                <DropdownMenuItem
                  key={languageOption}
                  onClick={() => setLanguage(languageOption)}
                >
                  {LANGUAGE_LABELS[languageOption]}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* `overflow-clip` so gutter does not render on top of focus ring */}
      <div className="overflow-clip rounded-sm transition focus-within:ring-2 focus-within:ring-blue-500">
        <CodeMirror
          value={value}
          onChange={onChange}
          onBlur={onBlur}
          extensions={extensions}
          theme={githubLightInit({
            settings: {
              fontFamily:
                language === "text" ? "var(--font-sans)" : "var(--font-mono)",
              fontSize: "var(--text-xs)",
              gutterBorder: "transparent",
              background: "transparent",
            },
          })}
          placeholder={placeholder}
          basicSetup={{
            // Line numbers
            lineNumbers: showLineNumbers && language !== "text",
            foldGutter: showLineNumbers && language !== "text",

            // Read-only mode
            autocompletion: !readOnly,
            searchKeymap: !readOnly,
            closeBrackets: !readOnly,
            dropCursor: !readOnly,
            allowMultipleSelections: !readOnly,
            highlightActiveLine: !readOnly,
            highlightActiveLineGutter: !readOnly,
          }}
        />
      </div>
    </div>
  );
};
