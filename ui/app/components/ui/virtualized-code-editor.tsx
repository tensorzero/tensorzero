import { useRef, useState, useEffect } from "react";
import { CodeEditor, type CodeEditorProps } from "./code-editor";
import { cn } from "~/utils/common";

/**
 * A virtualized wrapper around CodeEditor that renders a lightweight <pre>
 * placeholder when off-screen, swapping to the full CodeMirror editor when
 * the element enters the viewport.
 *
 * The placeholder text remains in the DOM, so Cmd+F browser search works
 * for off-screen content.
 */
export function VirtualizedCodeEditor(props: CodeEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        // Once visible, stay visible (don't unmount CodeMirror on scroll away)
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { rootMargin: "200px" }, // Pre-load when within 200px of viewport
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef}>
      {isVisible ? (
        <CodeEditor {...props} ariaLabel={props.ariaLabel} />
      ) : (
        <LightweightPlaceholder
          value={props.value}
          maxHeight={props.maxHeight}
          className={props.className}
        />
      )}
    </div>
  );
}

function LightweightPlaceholder({
  value,
  maxHeight = "400px",
  className,
}: {
  value?: string;
  maxHeight?: string;
  className?: string;
}) {
  return (
    <pre
      className={cn(
        "min-h-9 overflow-auto rounded-sm bg-gray-50 p-2 font-mono text-xs whitespace-pre-wrap",
        className,
      )}
      style={{ maxHeight }}
    >
      {value || ""}
    </pre>
  );
}
