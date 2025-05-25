import { SkeletonImage } from "~/components/inference/SkeletonImage";
import { Card, CardContent } from "~/components/ui/card";
import type {
  Input,
  ResolvedInputMessage,
  ResolvedInputMessageContent,
} from "~/utils/clickhouse/common";
import ImageBlock from "../inference/ImageBlock";

interface InputProps {
  input_messages: ResolvedInputMessage[];
  system: string | null;
}

function MessageContent({
  content,
}: {
  content: ResolvedInputMessageContent[];
}) {
  return (
    <div className="space-y-2">
      {content.map((block, blockIndex) => {
        switch (block.type) {
          case "text":
            return (
              <div key={blockIndex} className="whitespace-pre-wrap">
                <code className="text-sm">
                  {typeof block.value === "object"
                    ? JSON.stringify(block.value, null, 2)
                    : block.value}
                </code>
              </div>
            );
          case "tool_call":
            return (
              <div
                key={blockIndex}
                className="rounded bg-slate-100 p-2 dark:bg-slate-800"
              >
                <div className="font-medium">Tool: {block.name}</div>
                <pre className="mt-1 text-sm">{block.arguments}</pre>
              </div>
            );
          case "tool_result":
            return (
              <div
                key={blockIndex}
                className="rounded bg-slate-100 p-2 dark:bg-slate-800"
              >
                <div className="font-medium">Result from: {block.name}</div>
                <pre className="mt-1 text-sm">{block.result}</pre>
              </div>
            );
          case "file":
            if (block.file.mime_type.startsWith("image/")) {
              return <ImageBlock key={blockIndex} image={block} />;
            } else {
              return (
                <div key={blockIndex}>
                  <SkeletonImage error={`Unsupported file type: ${block.file.mime_type}`} />
                </div>
              )
            }
          case "file_error":
            return (
              <div key={blockIndex}>
                <SkeletonImage error="Failed to retrieve image." />
              </div>
            );
        }
      })}
    </div>
  );
}

function Message({ message }: { message: ResolvedInputMessage }) {
  return (
    <div className="space-y-1">
      <div className="font-medium text-slate-600 capitalize dark:text-slate-400">
        {message.role}
      </div>
      <MessageContent content={message.content} />
    </div>
  );
}

export default function Input({ input_messages, system }: InputProps) {
  return (
    <Card className="pt-6">
      <CardContent className="space-y-6">
        {system && (
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <div className="text-md mb-3 font-semibold text-slate-900 dark:text-slate-100">
              System
            </div>
            <pre className="overflow-x-auto p-4">
              <code className="text-sm">{system}</code>
            </pre>
          </div>
        )}

        <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
          <div className="text-md mb-3 font-semibold text-slate-900 dark:text-slate-100">
            Messages
          </div>
          <div className="space-y-4">
            {input_messages.map((message, index) => (
              <Message key={index} message={message} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
