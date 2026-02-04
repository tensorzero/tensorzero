/**
 * A small dot separator for use between text elements.
 * Commonly used between date/time, labels, or other inline content.
 */
export function DotSeparator() {
  return (
    <span
      className="bg-fg-muted inline-block h-0.5 w-0.5 rounded-full"
      aria-hidden="true"
    />
  );
}
