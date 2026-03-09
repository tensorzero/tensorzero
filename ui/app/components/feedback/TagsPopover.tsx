import { useState, useRef, useCallback, useEffect } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";

const T0_PREFIX = "tensorzero::";

function formatTagKey(key: string): string {
  if (key.startsWith(T0_PREFIX)) {
    return key.slice(T0_PREFIX.length);
  }
  return key;
}

function InternalBadge() {
  return (
    <span className="bg-bg-tertiary text-fg-tertiary shrink-0 rounded px-1 py-0.5 font-sans text-[10px] font-medium leading-none">
      T0
    </span>
  );
}

export function filterStringTags(
  tags: Record<string, unknown>,
): [string, string][] {
  return Object.entries(tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );
}

interface TagCellProps {
  displayText: string;
  fullText: string;
  className: string;
  tooltipSide: "left" | "right";
}

// Intentionally shows a tooltip only when displayText differs from fullText
// (e.g. tensorzero:: keys get their prefix stripped). For user tags and values
// where the two are identical, we skip the tooltip and dotted underline.
function TagCell({
  displayText,
  fullText,
  className,
  tooltipSide,
}: TagCellProps) {
  if (displayText === fullText) {
    return <span className={className}>{displayText}</span>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={`${className} underline decoration-dotted decoration-border underline-offset-2`}
        >
          {displayText}
        </span>
      </TooltipTrigger>
      <TooltipContent side={tooltipSide}>
        <span className="break-all font-mono text-xs">{fullText}</span>
      </TooltipContent>
    </Tooltip>
  );
}

interface TagsPopoverProps {
  tags: [string, string][];
}

export function TagsPopover({ tags }: TagsPopoverProps) {
  const [open, setOpen] = useState(false);
  const closeTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleClose = useCallback(() => {
    closeTimeout.current = setTimeout(() => setOpen(false), 150);
  }, []);

  const cancelClose = useCallback(() => {
    if (closeTimeout.current) {
      clearTimeout(closeTimeout.current);
      closeTimeout.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (closeTimeout.current) {
        clearTimeout(closeTimeout.current);
      }
    };
  }, []);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <span
          className="text-fg-tertiary cursor-default text-xs underline decoration-dotted decoration-border underline-offset-4"
          onPointerEnter={() => {
            cancelClose();
            setOpen(true);
          }}
          onPointerLeave={scheduleClose}
        >
          {tags.length} tag{tags.length !== 1 ? "s" : ""}
        </span>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="center"
        className="w-auto max-w-md p-3"
        onOpenAutoFocus={(e) => e.preventDefault()}
        onCloseAutoFocus={(e) => e.preventDefault()}
        onPointerEnter={cancelClose}
        onPointerLeave={scheduleClose}
      >
        <div className="space-y-1.5">
          {tags.map(([key, val]) => {
            const isInternal = key.startsWith(T0_PREFIX);
            return (
              <div
                key={key}
                className="flex items-center gap-2 font-mono text-xs"
              >
                <span className="flex w-40 shrink-0 items-center gap-1">
                  <TagCell
                    displayText={formatTagKey(key)}
                    fullText={key}
                    className="text-fg-secondary truncate"
                    tooltipSide="left"
                  />
                  {isInternal && <InternalBadge />}
                </span>
                <TagCell
                  displayText={val}
                  fullText={val}
                  className="text-fg-primary truncate"
                  tooltipSide="right"
                />
              </div>
            );
          })}
        </div>
      </PopoverContent>
    </Popover>
  );
}
