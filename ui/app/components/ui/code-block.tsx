import { clsx } from "clsx";

interface CodeBlockSharedProps {
  showLineNumbers?: boolean;
}

interface CodeBlockRawProps extends CodeBlockSharedProps {
  /**
   * A raw string of code.
   *
   * Because code strings can contain HTML entities, this should be
   * pre-sanitized before being passed to the component since it will be
   * unescaped.
   */
  raw: string;
  /**
   * A pre-formatted HTML string with the following structure:
   * ```html
   * <pre tabindex="0">
   *   <code>
   *     <span class="line">...</span>
   *   </code>
   * </pre>
   * ```
   *
   * This is the structure returned from Shiki's `codeToHtml` function, but this
   * can be provided with any other HTML renderer so long as each line is
   * wrapped in `<span class="line" />` for consistent styling.
   *
   * Because code strings can contain HTML entities, this should be
   * pre-sanitized before being passed to the component since it will be
   * unescaped.
   */
  html?: never;
}

interface CodeBlockHtmlProps extends CodeBlockSharedProps {
  /**
   * A pre-formatted HTML string with the following structure:
   * ```html
   * <pre tabindex="0">
   *   <code>
   *     <span class="line">...</span>
   *   </code>
   * </pre>
   * ```
   *
   * This is the structure returned from Shiki's `codeToHtml` function, but this
   * can be provided with any other HTML renderer so long as each line is
   * wrapped in `<span class="line" />` for consistent styling.
   *
   * Because code strings can contain HTML entities, this should be
   * pre-sanitized before being passed to the component since it will be
   * unescaped.
   */
  html: string;
  /**
   * A raw string of code.
   *
   * Because code strings can contain HTML entities, this should be
   * pre-sanitized before being passed to the component since it will be
   * unescaped.
   */
  raw?: never;
}

export type CodeBlockProps = CodeBlockRawProps | CodeBlockHtmlProps;

export function CodeBlock({
  html,
  raw,
  showLineNumbers = false,
}: CodeBlockProps) {
  const codeProps = {
    dangerouslySetInnerHTML: html ? { __html: html } : undefined,
    children: raw ? (
      <pre tabIndex={0}>
        <code dangerouslySetInnerHTML={{ __html: handleRawContent(raw) }} />
      </pre>
    ) : undefined,
  };

  return (
    <div
      className={clsx(
        "CodeBlock relative font-mono text-sm",
        showLineNumbers && "CodeBlock--with-line-numbers",
      )}
    >
      <div
        className={clsx(
          // <pre> styles
          "**:[pre]:!bg-bg-primary **:[pre]:max-w-none **:[pre]:shrink-0 **:[pre]:grow **:[pre]:overflow-auto **:[pre]:rounded-lg **:[pre]:outline-offset-1",
          // line numbers have their own left padding so that they stick to
          // the left border when scrolled
          showLineNumbers ? "*:p-5 *:pl-0" : "*:p-5",
          // <code> styles
          "**:[code]:text-fg-primary **:[code]:relative **:[code]:flex **:[code]:min-w-min **:[code]:flex-col **:[code]:gap-1.5 **:[code]:whitespace-pre",
          // <span class="line"> styles
          "**:[.line]:m-0 **:[.line]:flex **:[.line]:h-4.5 **:[.line]:flex-0 **:[.line]:p-0",
        )}
        {...codeProps}
      />
    </div>
  );
}

function handleRawContent(raw: string) {
  return raw
    .split("\n")
    .map((line) => `<span class="line">${line}</span>`)
    .join("\n");
}
