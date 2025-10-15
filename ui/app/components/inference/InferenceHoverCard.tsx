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
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "...";
}

function getInputPreview(inference: ParsedInferenceRow): string {
  // Get the first text message content
  const firstMessage = inference.input.messages?.[0];
  if (firstMessage?.content?.[0]) {
    const firstContent = firstMessage.content[0];
    if (firstContent.type === "text") {
      return firstContent.text;
    }
  }
  
  // Fallback to system message
  if (inference.input.system && typeof inference.input.system === "string") {
    return inference.input.system;
  }
  
  return "No text input found";
}

function getOutputPreview(inference: ParsedInferenceRow): string {
  if (inference.function_type === "chat") {
    const output = inference.output as any[];
    const firstBlock = output?.[0];
    if (firstBlock?.type === "text") {
      return firstBlock.text;
    }
  } else if (inference.function_type === "json") {
    const output = inference.output as any;
    if (output?.parsed) {
      return JSON.stringify(output.parsed);
    } else if (output?.raw) {
      return output.raw;
    }
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
            onHover?.();
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
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <div className="text-sm text-gray-500">Loading...</div>
          </div>
        ) : inference ? (
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
                {truncateText(getOutputPreview(inference), 120)}
              </div>
            </div>
            <div className="text-xs text-gray-400 pt-1 border-t">
              {inference.function_name} • {inference.variant_name} • {inference.processing_time_ms}ms
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-500">Failed to load inference details</div>
        )}
      </PopoverContent>
    </Popover>
  );
}