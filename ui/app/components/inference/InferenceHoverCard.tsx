import { useState, useEffect } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "~/components/ui/popover";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

interface InferenceHoverCardProps {
  children: React.ReactNode;
  inference: ParsedInferenceRow | null;
  isLoading?: boolean;
  onHover?: () => void;
}

function truncateText(text: string, maxLength: number = 100): string {
  return text.length <= maxLength ? text : text.slice(0, maxLength) + "...";
}

function getInputPreview(inference: ParsedInferenceRow): string {
  // Get the first text message content
  const firstMessage = inference.input.messages?.[0];
  const firstContent = firstMessage?.content?.[0];
  if (firstContent?.type === "text") {
    return firstContent.text;
  }

  if (typeof inference.input.system === "string") {
    return inference.input.system;
  }

  return "No text input found";
}

function getOutputPreview(inference: ParsedInferenceRow): string {
  if (inference.function_type === "chat") {
    const output = inference.output as any[];
    if (!output || output.length === 0) return "Empty output";

    // First pass: look for text
    for (const block of output) {
      if (block?.type === "text" && block.text) {
        return block.text;
      }
    }

    // Second pass: look for thought if no text found
    for (const block of output) {
      if (block?.type === "thought" && block.text) {
        return block.text;
      }
    }

    // Fallbacks
    const firstBlock = output[0];
    if (firstBlock?.type === "tool_call") {
      const toolName = firstBlock.name || firstBlock.raw_name || "unknown";
      return `Tool call: ${toolName}`;
    }
    if (firstBlock?.type) {
      return `Content type: ${firstBlock.type}`;
    }

    return "No readable content";
  }

  if (inference.function_type === "json") {
    const output = inference.output as any;
    if (!output || typeof output !== "object") return "No output available";

    // Prefer parsed JSON
    if ("parsed" in output && output.parsed != null) {
      try {
        return JSON.stringify(output.parsed);
      } catch {
        /* ignore and try raw */
      }
    }

    // Fallback to raw string
    if ("raw" in output && typeof output.raw === "string" && output.raw) {
      return output.raw;
    }

    if ("parsed" in output || "raw" in output) {
      return "Empty JSON output";
    }

    return "Invalid output format";
  }

  return "No output available";
}


export function InferenceHoverCard({ 
  children, 
  inference, 
  isLoading, 
  onHover 
}: InferenceHoverCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hasTriggeredLoad] = useState(false);

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <div
          onMouseEnter={() => {
            setIsOpen(true);
            if (onHover && !inference && !isLoading && !hasTriggeredLoad) {
              onHover();
            }
          }}
          onMouseLeave={() => setIsOpen(false)}
          className="inline-block"
        >
          {children}
        </div>
      </PopoverTrigger>
      <PopoverContent 
        className="w-80 p-3" 
        side="right" 
        align="start"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
      >
        {inference ? (
          <div className="space-y-3">
            <div>
              <div className="text-xs font-medium text-gray-500 mb-1">INPUT</div>
              <div className="text-sm bg-gray-50 p-2 rounded text-gray-800 font-mono">
                {truncateText(getInputPreview(inference), 120)}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-gray-500 mb-1">OUTPUT</div>
              <div className="text-sm bg-gray-50 p-2 rounded text-gray-800 font-mono">
                {truncateText(getOutputPreview(inference), 200)}
              </div>
            </div>
            <div className="text-xs text-gray-400 pt-1 border-t space-y-1">
              <div>
                {inference.function_name} • {inference.variant_name} • {inference.processing_time_ms}ms
              </div>
              <div >
                ID: {inference.id}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center py-4">
            <div className="text-sm text-gray-500">Loading...</div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}