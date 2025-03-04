import { Card, CardContent } from "~/components/ui/card";
import type { Input, InputMessage } from "~/utils/clickhouse/common";
import { SkeletonImage } from "./SkeletonImage";

interface InputProps {
  input: Input;
}

function MessageContent({ content }: { content: InputMessage["content"] }) {
  return (
    <div className="space-y-2">
      {content.map((block, index) => {
        switch (block.type) {
          case "text":
            return (
              <pre key={index} className="whitespace-pre-wrap">
                <code className="text-sm">
                  {typeof block.value === "object"
                    ? JSON.stringify(block.value, null, 2)
                    : block.value}
                </code>
              </pre>
            );
          case "tool_call":
            return (
              <div
                key={index}
                className="rounded bg-slate-100 p-2 dark:bg-slate-800"
              >
                <div className="font-medium">Tool: {block.name}</div>
                <pre className="mt-1 text-sm">{block.arguments}</pre>
              </div>
            );
          case "tool_result":
            return (
              <div
                key={index}
                className="rounded bg-slate-100 p-2 dark:bg-slate-800"
              >
                <div className="font-medium">Result from: {block.name}</div>
                <pre className="mt-1 text-sm">{block.result}</pre>
              </div>
            );
          case "image":
            return <SkeletonImage key={index} />;
        }
      })}
    </div>
  );
}

function Message({ message }: { message: InputMessage }) {
  return (
    <div className="space-y-1">
      <div className="text-md font-medium capitalize text-slate-600 dark:text-slate-400">
        {message.role}
      </div>
      <MessageContent content={message.content} />
    </div>
  );
}

export default function Input({ input }: InputProps) {
  return (
    <Card>
      <CardContent className="space-y-6 pt-6">
        {input.system && (
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <div className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
              System
            </div>
            <pre className="overflow-x-auto p-4">
              <code className="text-sm">
                {typeof input.system === "object"
                  ? JSON.stringify(input.system, null, 2)
                  : input.system}
              </code>
            </pre>
          </div>
        )}

        <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
          <div className="mb-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
            Messages
          </div>
          <div className="space-y-4">
            {input.messages.map((message, index) => (
              <Message key={index} message={message} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
