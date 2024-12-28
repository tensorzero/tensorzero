import { z } from "zod";
import { Textarea } from "~/components/ui/textarea";
import { useEffect, useState } from "react";

export const ProgressInfoSchema = z.discriminatedUnion("provider", [
  z.object({
    provider: z.literal("openai"),
    data: z.record(z.any()),
    estimatedCompletionTimestamp: z.number().optional(),
  }),
  z.object({
    provider: z.literal("fireworks"),
    data: z.union([z.string(), z.record(z.any())]),
    jobUrl: z.string(),
  }),
]);

export type ProgressInfo = z.infer<typeof ProgressInfoSchema>;

type ProgressEntry = {
  timestamp: Date;
  info: ProgressInfo;
};

export function ProgressIndicator({
  progressInfo,
}: {
  progressInfo?: ProgressInfo;
}) {
  const [history, setHistory] = useState<ProgressEntry[]>([]);

  // Add new progress info to history when it changes
  useEffect(() => {
    if (progressInfo) {
      setHistory((prev) => [
        ...prev,
        {
          timestamp: new Date(),
          info: progressInfo,
        },
      ]);
    }
  }, [progressInfo]);

  if (!progressInfo) {
    return null;
  }

  const getSerializedData = (entry: ProgressEntry) => {
    if (entry.info.provider === "fireworks") {
      return typeof entry.info.data === "string"
        ? entry.info.data
        : JSON.stringify(entry.info.data, null, 2);
    }
    return JSON.stringify(entry.info.data, null, 2);
  };

  const getTextAreaHeight = (text: string) => {
    const lineCount = text.split("\n").length;
    // Assuming each line is roughly 24px tall
    return Math.max(lineCount * 24, 128); // Minimum height of 128px
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg mt-4">
      <div className="mb-2 font-medium">Progress History</div>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {/* Reverse history array so most recent entries appear at the top */}
        {[...history].reverse().map((entry, index) => {
          const serializedData = getSerializedData(entry);
          const height = getTextAreaHeight(serializedData);

          return (
            <div
              key={index}
              className="border-b border-gray-200 pb-4 last:border-0"
            >
              <div className="text-sm text-gray-500 mb-1">
                {entry.timestamp.toLocaleTimeString()}
              </div>
              <Textarea
                value={serializedData}
                style={{ height: `${height}px` }}
                className="w-full resize-none bg-transparent border-none focus:ring-0"
                readOnly
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
