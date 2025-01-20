import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import ModelInput from "./ModelInput";

interface ModelInferenceItemProps {
  inference: ParsedModelInferenceRow;
}

export function ModelInferenceItem({ inference }: ModelInferenceItemProps) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <dt className="text-lg font-semibold">Model</dt>
          <dd>
            <Code>{inference.model_name}</Code>
          </dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">Model Provider</dt>
          <dd>
            <Code>{inference.model_provider_name}</Code>
          </dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">Input Tokens</dt>
          <dd>{inference.input_tokens}</dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">Output Tokens</dt>
          <dd>{inference.output_tokens}</dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">Response Time</dt>
          <dd>{inference.response_time_ms}ms</dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">TTFT</dt>
          <dd>{inference.ttft_ms ? `${inference.ttft_ms}ms` : "N/A"}</dd>
        </div>
        <div>
          <dt className="text-lg font-semibold">Timestamp</dt>
          <dd>{new Date(inference.timestamp).toLocaleString()}</dd>
        </div>
      </div>

      <ModelInput
        input_messages={inference.input_messages}
        system={inference.system}
      />

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(inference.output, null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Raw Request</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(JSON.parse(inference.raw_request), null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Raw Response</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(JSON.parse(inference.raw_response), null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
