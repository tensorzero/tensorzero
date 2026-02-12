import { Children, isValidElement } from "react";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  CodeEditor,
  type CodeEditorProps,
  type Language,
} from "~/components/ui/code-editor";
import { cn } from "~/utils/common";

// Common spacing for block elements
const BLOCK_SPACING = "mb-3 last:mb-0";

// Common table cell styling
const TABLE_CELL_BASE = "border-border border px-3 py-2";

/**
 * Map markdown language identifiers to CodeEditor languages.
 * Falls back to "text" for unsupported languages (no syntax highlighting).
 */
function mapLanguage(lang: string | undefined): Language {
  if (!lang) return "text";
  const normalized = lang.toLowerCase();
  switch (normalized) {
    case "json":
    case "jsonc":
      return "json";
    case "md":
    case "markdown":
      return "markdown";
    case "jinja":
    case "jinja2":
      return "jinja2";
    default:
      return "text"; // No syntax highlighting for unknown languages
  }
}

/**
 * Recursively extract text from React children.
 */
function getTextContent(node: React.ReactNode): string {
  if (node == null) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(getTextContent).join("");
  if (isValidElement(node)) {
    const props = node.props as { children?: React.ReactNode };
    return getTextContent(props.children);
  }
  return "";
}

/**
 * Extract text content and language from pre > code children.
 * react-markdown wraps code blocks as <pre><code className="language-xxx">...</code></pre>
 */
function extractCodeBlockInfo(children: React.ReactNode): {
  code: string;
  language: Language;
} {
  const child = Children.only(children);
  if (isValidElement(child) && child.type === "code") {
    const props = child.props as {
      className?: string;
      children?: React.ReactNode;
    };
    const lang = props.className?.replace("language-", "");
    const code = getTextContent(props.children).trimEnd();
    return { code, language: mapLanguage(lang) };
  }
  // Fallback: use text for monospace font without syntax highlighting
  return { code: getTextContent(children).trimEnd(), language: "text" };
}

/**
 * Read-only code block with copy button and word wrap toggle.
 * Reusable component for displaying code snippets consistently.
 */
export interface ReadOnlyCodeBlockProps
  extends Pick<CodeEditorProps, "className" | "maxHeight"> {
  code: string;
  language?: Language;
}

export function ReadOnlyCodeBlock({
  code,
  language = "text",
  maxHeight,
  className,
}: ReadOnlyCodeBlockProps) {
  return (
    <CodeEditor
      value={code}
      readOnly
      allowedLanguages={[language]}
      autoDetectLanguage={false}
      showLineNumbers={false}
      maxHeight={maxHeight}
      className={className}
    />
  );
}

/**
 * Markdown code block component using ReadOnlyCodeBlock.
 */
function MarkdownCodeBlock({ children }: { children?: React.ReactNode }) {
  if (!children) return null;
  const { code, language } = extractCodeBlockInfo(children);
  return (
    <div className={BLOCK_SPACING}>
      <ReadOnlyCodeBlock code={code} language={language} />
    </div>
  );
}

const components: Components = {
  // Headings
  h1: ({ children }) => (
    <h1 className="mt-6 mb-4 text-2xl font-semibold first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-5 mb-3 text-xl font-semibold first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-4 mb-2 text-lg font-semibold first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-3 mb-2 text-base font-semibold first:mt-0">{children}</h4>
  ),
  h5: ({ children }) => (
    <h5 className="mt-2 mb-1 text-sm font-semibold first:mt-0">{children}</h5>
  ),
  h6: ({ children }) => (
    <h6 className="mt-2 mb-1 text-sm font-semibold first:mt-0">{children}</h6>
  ),

  // Paragraphs
  p: ({ children }) => <p className={BLOCK_SPACING}>{children}</p>,

  // Lists - symmetric vertical margin for consistent spacing
  ul: ({ children }) => (
    <ul className="my-3 list-disc space-y-1.5 pl-6">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-3 list-decimal space-y-1.5 pl-6">{children}</ol>
  ),
  li: ({ children }) => <li>{children}</li>,

  // Inline code - slightly smaller than body text for visual balance
  code: ({ children }) => (
    <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs font-medium">
      {children}
    </code>
  ),

  // Code blocks - use CodeEditor for consistent UX
  pre: MarkdownCodeBlock,

  // Blockquotes
  blockquote: ({ children }) => (
    <blockquote
      className={cn("border-border border-l-4 pl-4 italic", BLOCK_SPACING)}
    >
      {children}
    </blockquote>
  ),

  // Links
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-fg-brand hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),

  // Strong and emphasis
  strong: ({ children }) => (
    <strong className="font-semibold">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,

  // Horizontal rule
  hr: () => <hr className="border-border my-4" />,

  // Tables
  table: ({ children }) => (
    <div className={cn("overflow-x-auto", BLOCK_SPACING)}>
      <table className="border-border min-w-full border-collapse border">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr className="border-border border-b">{children}</tr>,
  th: ({ children }) => (
    <th className={cn(TABLE_CELL_BASE, "text-left font-semibold")}>
      {children}
    </th>
  ),
  td: ({ children }) => <td className={TABLE_CELL_BASE}>{children}</td>,
};

interface MarkdownProps {
  children: string;
  className?: string;
  /** Optional extra remark plugins to apply (e.g. UUID link processing). */
  remarkPlugins?: React.ComponentProps<typeof ReactMarkdown>["remarkPlugins"];
  /** Optional extra component overrides merged on top of the defaults. */
  extraComponents?: Components;
}

export function Markdown({
  children,
  className,
  remarkPlugins: extraRemarkPlugins,
  extraComponents,
}: MarkdownProps) {
  const mergedComponents = extraComponents
    ? { ...components, ...extraComponents }
    : components;
  return (
    <div className={cn("text-fg-secondary text-sm", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, ...(extraRemarkPlugins ?? [])]}
        components={mergedComponents}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
