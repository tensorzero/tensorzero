import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCopy } from "~/hooks/use-copy";
import { useLocalStorage } from "~/hooks/use-local-storage";
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
  X,
} from "lucide-react";
import type { JsonValue } from "~/types/tensorzero";
import debounce from "lodash-es/debounce";

export type Language = "json" | "markdown" | "jinja2" | "text";

/** Try to format the given string/object if it's JSON. Passthrough gracefully if it's not JSON. */
export function useFormattedJson(initialValue: string | JsonValue): string {
  return useMemo(() => {
    // If it's already a string
    if (typeof initialValue === "string") {
      try {
        // Only attempt to parse/format if it looks like JSON
        const trimmed = initialValue.trim();
        if (
          (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
          (trimmed.startsWith("[") && trimmed.endsWith("]"))
        ) {
          const parsed = JSON.parse(trimmed);
          return JSON.stringify(parsed, null, 2);
        }
      } catch {
        // Not valid JSON, return as-is
      }
      return initialValue;
    }

    // If it's an object/array/other JSON value, format it
    try {
      return JSON.stringify(initialValue, null, 2);
    } catch {
      // Fallback for circular references or other stringify errors
      return String(initialValue);
    }
  }, [initialValue]);
}

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
  /** We should generally set a maxHeight to improve performance for large documents. */
  maxHeight?: string;
  /** Aria label for accessibility */
  ariaLabel?: string;
}

const LANGUAGE_EXTENSIONS = {
  json: [json()],
  markdown: [markdown()],
  jinja2: [StreamLanguage.define(jinja2)],
  text: [],
} as const;

// Shared theme for focus and gutter styles (created once, shared across all instances)
const CUSTOM_EDITOR_THEME = EditorView.theme({
  "&.cm-focused": {
    outline: "none !important",
  },
  ".cm-gutters": {
    fontFamily: "var(--font-mono) !important",
  },
});

// Pre-computed theme variants (only 2 needed: mono vs sans font)
const THEME_MONO = githubLightInit({
  settings: {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    gutterBorder: "transparent",
    background: "transparent",
  },
});
const THEME_SANS = githubLightInit({
  settings: {
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-xs)",
    gutterBorder: "transparent",
    background: "transparent",
  },
});

// Cache for extension combinations (max 16 combinations: 4 languages × 2 wordWrap × 2 readOnly)
const extensionCache = new Map<string, Extension[]>();

function getExtensions(
  language: Language,
  wordWrap: boolean,
  readOnly: boolean,
): Extension[] {
  const key = `${language}-${wordWrap}-${readOnly}`;
  let exts = extensionCache.get(key);
  if (!exts) {
    exts = [...LANGUAGE_EXTENSIONS[language], CUSTOM_EDITOR_THEME];
    if (wordWrap) exts.push(EditorView.lineWrapping);
    if (readOnly) exts.push(EditorView.editable.of(false));
    extensionCache.set(key, exts);
  }
  return exts;
}

// Cache for basicSetup combinations (max 4 combinations: 2 showLineNumbers × 2 readOnly)
const basicSetupCache = new Map<string, object>();

function getBasicSetup(showLineNumbers: boolean, readOnly: boolean) {
  const key = `${showLineNumbers}-${readOnly}`;
  let setup = basicSetupCache.get(key);
  if (!setup) {
    setup = {
      lineNumbers: showLineNumbers,
      foldGutter: showLineNumbers,
      autocompletion: !readOnly,
      searchKeymap: !readOnly,
      closeBrackets: !readOnly,
      dropCursor: !readOnly,
      allowMultipleSelections: !readOnly,
      highlightActiveLine: !readOnly,
      highlightActiveLineGutter: !readOnly,
    };
    basicSetupCache.set(key, setup);
  }
  return setup;
}

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

const DEFAULT_WORD_WRAP_LANGUAGES: Language[] = [
  "text",
  "json",
  "jinja2",
  "markdown",
];

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
  maxHeight = "400px",
  ariaLabel,
}) => {
  // Internal state for semi-uncontrolled mode
  const [internalValue, setInternalValue] = useState(value);

  // Track external callbacks and the latest flushed/pending values
  const onChangeRef = useRef(onChange);
  const onBlurRef = useRef(onBlur);
  useEffect(() => {
    onChangeRef.current = onChange;
    onBlurRef.current = onBlur;
  }, [onChange, onBlur]);

  const committedValueRef = useRef(value);
  const pendingValueRef = useRef(value);

  const flushPending = useCallback(() => {
    const next = pendingValueRef.current;
    if (next !== committedValueRef.current) {
      committedValueRef.current = next;
      onChangeRef.current?.(next);
    }
  }, []);

  const debouncedFlush = useMemo(
    () =>
      debounce(
        () => {
          flushPending();
        },
        100,
        { leading: true, trailing: true },
      ),
    [flushPending],
  );

  // Sync external value changes to internal state and cancel pending debounces
  useEffect(() => {
    setInternalValue(value);
    pendingValueRef.current = value;
    committedValueRef.current = value;
    debouncedFlush.cancel();
  }, [value, debouncedFlush]);

  // Flush pending changes on unmount to prevent data loss
  useEffect(() => {
    return () => {
      debouncedFlush.flush();
      debouncedFlush.cancel();
    };
  }, [debouncedFlush]);

  // Handle internal value changes
  const handleChange = useCallback(
    (val: string) => {
      setInternalValue(val);
      pendingValueRef.current = val;
      if (!onChangeRef.current) {
        committedValueRef.current = val;
        return;
      }
      debouncedFlush();
    },
    [debouncedFlush],
  );

  // Handle blur: cancel debounce and flush immediately
  const handleBlur = useCallback(() => {
    debouncedFlush.flush();
    debouncedFlush.cancel();
    onBlurRef.current?.();
  }, [debouncedFlush]);

  // Update language when value changes if auto-detection is enabled
  const [language, setLanguage] = useState<Language>(() =>
    autoDetectLanguage ? detectLanguage(value) : allowedLanguages[0],
  );

  // Handle wrapping behavior
  const [wordWrap, setWordWrap] = useLocalStorage(
    "word-wrap",
    DEFAULT_WORD_WRAP_LANGUAGES.includes(language),
  );
  const toggleWordWrap = useCallback(() => {
    setWordWrap((wrap) => !wrap);
  }, [setWordWrap]);
  const { copy, didCopy, isCopyAvailable } = useCopy();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Use cached extensions (shared across all CodeEditor instances)
  const extensions = getExtensions(language, wordWrap, readOnly);

  const buttonClassName =
    "flex h-6 w-6 cursor-pointer items-center justify-center p-3 text-xs";

  // Use pre-computed theme (only 2 variants exist)
  const theme = language === "text" ? THEME_SANS : THEME_MONO;

  // Use cached basicSetup (shared across all CodeEditor instances)
  const basicSetup = getBasicSetup(showLineNumbers, readOnly);

  return (
    // `min-width: 0` If within a grid parent, prevent editor from overflowing its grid cell and force horizontal scrolling
    <div className={cn("group relative isolate min-w-0 rounded-sm", className)}>
      <div className="absolute top-1 right-1 z-10 flex gap-1.5 opacity-0 transition-opacity duration-200 group-hover:opacity-100 focus-within:opacity-100">
        <Button
          variant="secondary"
          size="iconSm"
          onClick={() => copy(internalValue)}
          className={buttonClassName}
          disabled={!mounted || !isCopyAvailable}
          title={didCopy ? "Copied!" : "Copy to clipboard"}
        >
          {didCopy ? (
            <CheckCheckIcon className="h-2 w-2" />
          ) : (
            <ClipboardIcon className="h-2 w-2" />
          )}
        </Button>
        <Button
          variant="secondary"
          size="iconSm"
          onClick={() => toggleWordWrap()}
          aria-pressed={wordWrap}
          className={buttonClassName}
          title="Toggle word wrap"
        >
          <span className="relative flex h-full w-full items-center justify-center">
            <WrapTextIcon className="absolute top-1/2 left-1/2 z-10 h-3 w-3 -translate-x-1/2 -translate-y-1/2" />
            {/* If disabled, show an X icon, larger and on top of the wrap icon */}
            {wordWrap ? null : (
              <X
                className="absolute top-1/2 left-1/2 z-20 !h-7 !w-7 -translate-x-1/2 -translate-y-1/2"
                strokeWidth={1}
              />
            )}
          </span>
        </Button>

        {allowedLanguages.length > 1 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="secondary"
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
      <div className="overflow-clip rounded-sm bg-gray-50 transition focus-within:ring-2 focus-within:ring-blue-500">
        <CodeMirror
          value={internalValue}
          onChange={handleChange}
          onBlur={handleBlur}
          extensions={extensions}
          theme={theme}
          placeholder={placeholder}
          basicSetup={basicSetup}
          maxHeight={maxHeight}
          className="min-h-9 overflow-auto"
          aria-label={ariaLabel}
        />
      </div>
    </div>
  );
};
