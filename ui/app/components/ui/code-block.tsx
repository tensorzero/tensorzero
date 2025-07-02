"use client";
import * as React from "react";
import { clsx } from "clsx";
import { Button, type ButtonProps } from "./button";
import { DummyCheckbox } from "./checkbox";

interface CodeBlockSharedProps {
  showLineNumbers?: boolean;
  /** @default true */
  showWrapToggle?: boolean;
  /** @default true */
  padded?: boolean;
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
  padded = true,
  showWrapToggle = true,
}: CodeBlockProps) {
  const [wrapLinesState, setWrapLines] = React.useState(false);

  const wrapLines = showWrapToggle ? wrapLinesState : false;

  const codeProps = html
    ? { dangerouslySetInnerHTML: { __html: html } }
    : {
        children: raw ? (
          <pre tabIndex={0}>
            <code dangerouslySetInnerHTML={{ __html: handleRawContent(raw) }} />
          </pre>
        ) : undefined,
      };

  return (
    <div
      data-show-line-numbers={showLineNumbers || undefined}
      data-padded={padded || undefined}
      data-wrap-lines={wrapLines || undefined}
      className={clsx("CodeBlock", "group relative isolate")}
    >
      {showWrapToggle && (
        <div className="pointer-events-none absolute top-0 right-0 z-10 flex justify-end p-2 opacity-0 group-hover:opacity-100 focus-within:opacity-100">
          <ToggleButton
            type="button"
            toggled={wrapLines}
            onToggle={setWrapLines}
            className="pointer-events-auto pl-2"
            size="sm"
            variant="outline"
          >
            <DummyCheckbox checked={wrapLines} />
            <span>Wrap</span>
          </ToggleButton>
        </div>
      )}
      <div
        className={clsx(
          "font-mono text-sm",
          // <pre> styles
          "**:[pre]:!bg-bg-primary **:[pre]:max-w-none **:[pre]:shrink-0 **:[pre]:grow **:[pre]:overflow-auto **:[pre]:rounded-lg **:[pre]:outline-offset-1",
          padded &&
            (showLineNumbers
              ? // line numbers have their own left padding so that they stick to
                // the left border when scrolled
                "*:p-5 *:pl-0"
              : "*:p-5"),
          // <code> styles
          "**:[code]:text-fg-primary **:[code]:relative **:[code]:flex **:[code]:min-w-min **:[code]:flex-col **:[code]:gap-1.5",
          wrapLines
            ? "**:[code]:whitespace-pre-wrap"
            : "**:[code]:whitespace-pre",
          // <span class="line"> styles
          "**:[.line]:m-0 **:[.line]:flex **:[.line]:h-4.5 **:[.line]:flex-0 **:[.line]:p-0",
        )}
        {...codeProps}
      />
    </div>
  );
}

function ToggleButton({
  toggled,
  onToggle,
  ...props
}: Omit<ButtonProps, "onToggle"> & {
  toggled: boolean;
  onToggle: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  return (
    <Button
      type="button"
      aria-pressed={toggled}
      {...props}
      onClick={(event) => {
        props.onClick?.(event);
        if (!event.defaultPrevented) {
          onToggle?.((toggled) => !toggled);
        }
      }}
    />
  );
}

function handleRawContent(raw: string) {
  return raw
    .split("\n")
    .map((line) => `<span class="line">${line}</span>`)
    .join("\n");
}
