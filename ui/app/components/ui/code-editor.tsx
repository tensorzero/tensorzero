import { useEffect, useMemo, useState } from "react";
import { useCopy } from "~/hooks/use-copy";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { StreamLanguage } from "@codemirror/language";
import { jinja2 } from "@codemirror/legacy-modes/mode/jinja2";
import { githubLightInit, githubDarkInit } from "@uiw/codemirror-theme-github";
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

export type Language = "json" | "markdown" | "jinja2" | "text"; // TODO satisfies something?

export interface CodeEditorProps {
  value?: string;
  onChange?: (value: string) => void;

  /** Languages to show in selector. If auto detect is disabled, defaults to the first one. */
  allowedLanguages?: [Language, ...Language[]];
  autoDetectLanguage?: boolean;

  readOnly?: boolean;
  showLineNumbers?: boolean;

  height?: string;
  minHeight?: string;
  maxHeight?: string;
  placeholder?: string;
  theme?: "light" | "dark"; // TODO remove dark?

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

export const CodeEditor: React.FC<CodeEditorProps> = ({
  value = "",
  onChange,
  allowedLanguages = ["text", "markdown", "json", "jinja2"],
  readOnly = false,
  showLineNumbers = true,

  // TODO Change this
  // height,
  // minHeight = "100px",
  // maxHeight,
  placeholder = "Enter your code here...", // TODO Do we want this?
  className,
  autoDetectLanguage = allowedLanguages.length > 1,
  theme = "light",
}) => {
  const [language, setLanguage] = useState<Language>(() =>
    autoDetectLanguage ? detectLanguage(value) : allowedLanguages[0],
  );

  const [wordWrap, setWordWrap] = useState(false);
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
    <div
      className={cn(
        "group relative rounded-sm transition focus-within:ring-2 focus-within:ring-blue-500",
        className,
      )}
    >
      {/* Note: for focus-within to work properly, `DropdownMenu` cannot render with a portal or the buttons will disappear when the dropdown is open */}
      <div className="absolute top-1 right-1 z-10 flex gap-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100 focus-within:opacity-100">
        {isCopyAvailable && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => copy(value)}
            className="flex h-6 w-6 p-3 text-xs"
            title={didCopy ? "Copied!" : "Copy to clipboard"}
          >
            {didCopy ? (
              <CheckCheckIcon className="h-2 w-2" />
            ) : (
              <ClipboardIcon className="h-2 w-2" />
            )}
          </Button>
        )}

        <Button
          variant={wordWrap ? "default" : "outline"}
          size="sm"
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

      {/* Better way to do this? */}
      <style>{`.cm-editor.cm-focused { outline: none; }`}</style>

      <CodeMirror
        value={value}
        onChange={onChange}
        extensions={extensions}
        theme={(theme === "dark" ? githubDarkInit : githubLightInit)({
          settings: {
            fontFamily:
              language === "text" ? "var(--font-sans)" : "var(--font-mono)",
            fontSize: "var(--text-xs)",
            gutterBorder: "transparent",

            background: "transparent", // TODO Only if readOnly?

            // Note: if this is transparent, line numbers are hard to read when scrolling horizontally
            // gutterBackground: "transparent",
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
        // className={className}

        //   // TODO does this work
        // style={{
        //   ...(height && { height }),
        //   ...(minHeight && { minHeight }),
        //   ...(maxHeight && { maxHeight }),
        // }}
      />
    </div>
  );
};

export const JsonEditor: React.FC<{
  value: string | object;
  setValue?: React.Dispatch<React.SetStateAction<string>>;
  className?: string;
}> = ({ value, setValue, className, ...props }) => {
  const [hasBeenModified, setHasBeenModified] = useState(false);
  const [isValidJson, setIsValidJson] = useState(true);

  // Convert object to string and prettify if needed
  const stringValue = useMemo(() => {
    if (typeof value === "object" && value !== null) {
      return JSON.stringify(value, null, 2);
    }

    if (typeof value === "string" && value.trim() && !hasBeenModified) {
      try {
        const parsed = JSON.parse(value);
        return JSON.stringify(parsed, null, 2);
      } catch {
        // If it's not valid JSON, return as-is
        return value;
      }
    }

    return value;
  }, [value, hasBeenModified]);

  // TODO is this duplicative with useEffect...?
  const handleChange = (newValue: string) => {
    setHasBeenModified(true);

    // Validate JSON
    if (newValue.trim()) {
      try {
        JSON.parse(newValue);
        setIsValidJson(true);
      } catch {
        setIsValidJson(false);
      }
    } else {
      setIsValidJson(true); // Empty is considered valid
    }

    setValue?.(newValue);
  };

  // Reset modification flag and validate when value changes externally
  useEffect(() => {
    if (typeof value === "object") {
      setHasBeenModified(false);
      setIsValidJson(true); // Objects are always valid
    } else if (typeof value === "string" && value.trim()) {
      try {
        JSON.parse(value);
        setIsValidJson(true);
      } catch {
        setIsValidJson(false);
      }
    } else {
      setIsValidJson(true); // Empty is valid
    }
  }, [value]);

  return (
    <CodeEditor
      {...props}
      value={stringValue}
      onChange={handleChange}
      allowedLanguages={["json"]}
      autoDetectLanguage={false}
      className={cn(
        "max-w-full min-w-80",
        !isValidJson && "focus-within:ring-red-500",
        className,
      )}
      readOnly={false}
    />
  );
};
