import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import type {
  JsonInferenceOutput,
  ContentBlockOutput,
} from "~/utils/clickhouse/common";

interface OutputProps {
  output: JsonInferenceOutput | ContentBlockOutput[];
}

function isJsonInferenceOutput(
  output: OutputProps["output"],
): output is JsonInferenceOutput {
  return "raw" in output;
}

function renderContentBlock(block: ContentBlockOutput, index: number) {
  switch (block.type) {
    case "text":
      return (
        <div key={index} className="whitespace-pre-wrap">
          {block.text}
        </div>
      );
    case "tool_call":
      return (
        <div key={index} className="my-2 rounded-md bg-muted/50 p-2 font-mono">
          <div className="font-semibold">Tool: {block.name}</div>
          <pre className="overflow-x-auto">
            <code>{JSON.stringify(block.arguments, null, 2)}</code>
          </pre>
        </div>
      );
  }
}

export default function Output({ output }: OutputProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Output</CardTitle>
      </CardHeader>
      <CardContent>
        {isJsonInferenceOutput(output) ? (
          <div className="space-y-4">
            {output.parsed && (
              <div className="rounded-md bg-muted p-4">
                <h3 className="mb-2 text-sm font-medium">Parsed Output</h3>
                <pre className="overflow-x-auto">
                  <code className="text-sm">
                    {JSON.stringify(output.parsed, null, 2)}
                  </code>
                </pre>
              </div>
            )}
            <div className="rounded-md bg-muted p-4">
              <h3 className="mb-2 text-sm font-medium">Raw Output</h3>
              <pre className="overflow-x-auto">
                <code className="text-sm">{output.raw}</code>
              </pre>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            {output.map((block, index) => renderContentBlock(block, index))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
