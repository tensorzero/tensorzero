import { Children, isValidElement } from "react";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Code } from "~/components/ui/code";
import { CodeEditor, type Language } from "~/components/ui/code-editor";
import { cn } from "~/utils/common";

/**
 * Map markdown language identifiers to CodeEditor languages.
 * Falls back to "text" for unsupported languages.
 */
function mapLanguage(lang: string | undefined): Language {
  if (!lang) return "text";
  const normalized = lang.toLowerCase();
  if (normalized === "json" || normalized === "jsonc") return "json";
  if (normalized === "md" || normalized === "markdown") return "markdown";
  if (normalized === "jinja" || normalized === "jinja2") return "jinja2";
  return "text";
}

/**
 * Extract text content and language from pre > code children.
 */
function extractCodeBlockInfo(children: React.ReactNode): {
  code: string;
  language: Language;
} {
  // react-markdown wraps code blocks as <pre><code className="language-xxx">...</code></pre>
  const child = Children.only(children);
  if (isValidElement(child) && child.type === "code") {
    const className = (child.props as { className?: string }).className;
    const lang = className?.replace("language-", "");
    const code = String(
      (child.props as { children?: React.ReactNode }).children || "",
    );
    // Remove trailing newline that markdown adds
    return { code: code.replace(/\n$/, ""), language: mapLanguage(lang) };
  }
  return { code: String(children), language: "text" };
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
  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,

  // Lists
  ul: ({ children }) => (
    <ul className="mb-3 list-disc space-y-1 pl-6 last:mb-0">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-3 list-decimal space-y-1 pl-6 last:mb-0">{children}</ol>
  ),
  li: ({ children }) => <li>{children}</li>,

  // Inline code - reuse existing Code component
  code: ({ className, children }) => {
    // Code blocks are handled by pre, this is only for inline code
    const isCodeBlock = className?.startsWith("language-");
    if (isCodeBlock) {
      // Let pre handle it
      return (
        <code className={cn("block whitespace-pre-wrap", className)}>
          {children}
        </code>
      );
    }
    return <Code>{children}</Code>;
  },

  // Code blocks - use CodeEditor for consistent UX
  pre: ({ children }) => {
    const { code, language } = extractCodeBlockInfo(children);
    return (
      <div className="mb-3 last:mb-0">
        <CodeEditor
          value={code}
          readOnly
          allowedLanguages={[language]}
          autoDetectLanguage={false}
          showLineNumbers={false}
          maxHeight="300px"
        />
      </div>
    );
  },

  // Blockquotes
  blockquote: ({ children }) => (
    <blockquote className="border-border text-fg-secondary mb-3 border-l-4 pl-4 italic last:mb-0">
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
    <div className="mb-3 overflow-x-auto last:mb-0">
      <table className="border-border min-w-full border-collapse border">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr className="border-border border-b">{children}</tr>,
  th: ({ children }) => (
    <th className="border-border border px-3 py-2 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border-border border px-3 py-2">{children}</td>
  ),
};

interface MarkdownProps {
  children: string;
  className?: string;
}

export function Markdown({ children, className }: MarkdownProps) {
  return (
    <div className={cn("text-fg-secondary text-sm", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
