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
        <div key={index} className="rounded-md bg-muted p-4">
          <h3 className="mb-2 text-lg font-medium">Text</h3>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">{block.text}</code>
          </pre>
        </div>
      );
    case "tool_call":
      return (
        <div key={index} className="rounded-md bg-muted p-4">
          <h3 className="mb-2 text-lg font-medium">Tool: {block.name}</h3>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">
              {JSON.stringify(block.arguments, null, 2)}
            </code>
          </pre>
        </div>
      );
  }
}

export default function Output({ output }: OutputProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Output</CardTitle>
      </CardHeader>
      <CardContent>
        {isJsonInferenceOutput(output) ? (
          <div className="space-y-4">
            {output.parsed && (
              <div className="rounded-md bg-muted p-4">
                <h3 className="mb-2 text-lg font-medium">Parsed Output</h3>
                <pre className="overflow-x-auto whitespace-pre-wrap break-words">
                  <code className="text-sm">
                    {JSON.stringify(output.parsed, null, 2)}
                  </code>
                </pre>
              </div>
            )}
            <div className="rounded-md bg-muted p-4">
              <h3 className="mb-2 text-lg font-medium">Raw Output</h3>
              <pre className="overflow-x-auto whitespace-pre-wrap break-words">
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
