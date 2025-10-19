import { useState, useEffect } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "~/components/ui/popover";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

interface InferenceHoverCardProps {
  children: React.ReactNode;
  inference: ParsedInferenceRow | null;
  isLoading?: boolean;
  onHover?: () => void;
}

type ChatOutputBlock =
  | { type: "text"; text: string }
  | { type: "thought"; text: string }
  | { type: "tool_call"; name?: string; raw_name?: string }
  | { type: string; [key: string]: any }; // fallback for unknown types

type JsonOutput = {
  parsed?: any;
  raw?: string;
  [key: string]: any;
};

function truncateText(text: string, maxLength: number): string {
  return text.length <= maxLength ? text : text.slice(0, maxLength) + "...";
}

function getInputPreview(inference: ParsedInferenceRow): string {
  const message = inference.input.messages?.[0];
  const content = message?.content?.[0];
  if (content?.type === "text") {
    return content.text;
  }

  if (typeof inference.input.system === "string") {
    return inference.input.system;
  }

  return "No text input found";
}

function getOutputPreview(inference: ParsedInferenceRow): string {
  if (inference.function_type === "chat") {
    const output = inference.output as ChatOutputBlock[];
    if (!output || output.length === 0) {
      return "Empty output";
    }

    for (const block of output) {
      if (block?.type === "text" && block.text) {
        return block.text;
      }
    }

    // Fallbacks
    for (const block of output) {
      if (block?.type === "thought" && block.text) {
        return block.text;
      }
    }

    const block = output[0];
    if (block?.type === "tool_call") {
      const toolName = block.name || block.raw_name || "unknown";
      return `Tool call: ${toolName}`;
    }
    if (block?.type) {
      return `Content type: ${block.type}`;
    }

    return "No readable content";
  }

  if (inference.function_type === "json") {
    const output = inference.output as JsonOutput;
    if (!output || typeof output !== "object") return "No output available";

    // Prefer parsed JSON
    if ("parsed" in output && output.parsed != null) {
      try {
        return JSON.stringify(output.parsed);
      } catch {
        // If stringify fails, fall back to raw
      }
    }
    
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

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <div
          onMouseEnter={() => {
            setIsOpen(true);
            if (onHover && !inference && !isLoading) {
              onHover();
            }
          }}
          onMouseLeave={() => {
            setIsOpen(false);
          }}
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
              <div>
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