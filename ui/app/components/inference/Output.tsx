import { Badge } from "~/components/ui/badge";
import { Card, CardContent } from "~/components/ui/card";
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
          <Badge className="mb-2">Text</Badge>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">{block.text}</code>
          </pre>
        </div>
      );
    case "tool_call":
      return (
        <div key={index} className="rounded-md bg-muted p-4">
          <Badge className="mb-2">Tool: {block.name}</Badge>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">
              {JSON.stringify(block.arguments, null, 2)}
            </code>
          </pre>
        </div>
      );
  }
}

export function OutputContent({ output }: OutputProps) {
  return isJsonInferenceOutput(output) ? (
    <div className="space-y-4">
      {output.parsed && (
        <div className="rounded-md bg-muted p-4">
          <Badge className="mb-2">Parsed Output</Badge>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words">
            <code className="text-sm">
              {JSON.stringify(output.parsed, null, 2)}
            </code>
          </pre>
        </div>
      )}
      <div className="rounded-md bg-muted p-4">
        <Badge className="mb-2">Raw Output</Badge>
        <pre className="overflow-x-auto whitespace-pre-wrap break-words">
          <code className="text-sm">{output.raw}</code>
        </pre>
      </div>
    </div>
  ) : (
    <div className="space-y-2">
      {output.map((block, index) => renderContentBlock(block, index))}
    </div>
  );
}

export default function Output({ output }: OutputProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <OutputContent output={output} />
      </CardContent>
    </Card>
  );
}
