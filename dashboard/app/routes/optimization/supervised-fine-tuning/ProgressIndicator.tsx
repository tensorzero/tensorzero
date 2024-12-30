import { z } from "zod";
import { Textarea } from "~/components/ui/textarea";
import { useEffect, useState } from "react";
import { CountdownTimer } from "./CountdownTimer";

export const ProgressInfoSchema = z.discriminatedUnion("provider", [
  z.object({
    provider: z.literal("openai"),
    data: z.record(z.any()),
    estimatedCompletionTimestamp: z.number().optional(), // needs to be in ms (not OpenAI default)
    jobUrl: z.string(),
  }),
  z.object({
    provider: z.literal("fireworks"),
    data: z.union([z.string(), z.record(z.any())]),
    jobUrl: z.string(),
  }),
  z.object({
    provider: z.literal("error"),
    data: z.object({ message: z.string() }),
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
    if (typeof entry.info.data === "string") {
      return entry.info.data;
    }
    return JSON.stringify(entry.info.data, null, 2);
  };

  // TODO: fix this so there's not extra space at the bottom
  // All the suggestions I've gotten from LLMs & StackOverflow are ridiculous
  // and I hope there's something sensible for this
  const getTextAreaHeight = (text: string) => {
    const lineCount = text.split("\n").length;
    return Math.max(lineCount * 24, 128);
  };

  const formatEstimatedTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const getEntryClassName = (entry: ProgressEntry) => {
    if (entry.info.provider === "error") {
      return "border-red-300 bg-red-50";
    }
    return "border-gray-200";
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg mt-4">
      <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col justify-center">
          <div className="mb-2 font-medium">Job URL</div>
          <a
            href={progressInfo.jobUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="w-full text-blue-600 hover:text-blue-800"
          >
            <Textarea
              value={progressInfo.jobUrl}
              className="w-full resize-none bg-white cursor-pointer hover:bg-gray-50"
              readOnly
            />
          </a>
        </div>

        {progressInfo.provider === "openai" &&
          progressInfo.estimatedCompletionTimestamp && (
            <div className="flex flex-col justify-center">
              <div className="mb-2 font-medium">Estimated Completion</div>
              <div className="bg-white p-2 rounded-md">
                <div>
                  {formatEstimatedTime(
                    progressInfo.estimatedCompletionTimestamp,
                  )}
                </div>
                <CountdownTimer
                  endTime={progressInfo.estimatedCompletionTimestamp}
                />
              </div>
            </div>
          )}
      </div>

      <div className="mb-2 font-medium">Progress History</div>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {[...history].reverse().map((entry, index) => {
          const serializedData = getSerializedData(entry);
          const height = getTextAreaHeight(serializedData);

          return (
            <div
              key={index}
              className={`border rounded-md p-3 ${getEntryClassName(
                entry,
              )} last:border-b`}
            >
              <div className="flex justify-between items-center mb-1">
                <div className="text-sm text-gray-500">
                  {entry.timestamp.toLocaleTimeString()}
                </div>
                {entry.info.provider === "error" && (
                  <div className="text-sm font-medium text-red-600">Error</div>
                )}
              </div>
              <Textarea
                value={serializedData}
                style={{ height: `${height}px` }}
                className={`w-full resize-none border-none focus:ring-0 ${
                  entry.info.provider === "error"
                    ? "bg-red-50 text-red-700"
                    : "bg-transparent"
                }`}
                readOnly
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
