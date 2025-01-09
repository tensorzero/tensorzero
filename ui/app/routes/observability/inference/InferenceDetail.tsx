import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import { Badge } from "~/components/ui/badge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";

// Mock data for demonstration
const inferenceData = {
  id: "0000-0000-0000-0000",
  function: "write_haiku",
  variant: "baseline",
  variantType: "chat_completion",
  episodeId: "1111-1111-1111-1111",
  timestamp: "2025-01-06T13:16:21Z",
  processingTime: "250ms",
  tags: {
    client_id: "2557",
    author: "Gabriel",
    namespace: "megumin",
  },
  input: {
    prompt: "Write a haiku about artificial intelligence",
  },
  output: {
    haiku:
      "Silicon dreams wake\nNeural networks pulse with thought\nHuman-made conscience",
  },
  inferenceParameters: {
    temperature: 0.7,
    max_tokens: 50,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  },
  toolParameters: {
    tool_name: "haiku_generator",
    version: "1.0.3",
    settings: {
      syllable_pattern: [5, 7, 5],
      theme: "technology",
    },
  },
  modelInferences: [
    {
      id: "9876-5432-1098-7654",
      model: "my_model",
      modelProvider: "my_provider",
      timestamp: "2025-01-06T13:16:21.500Z",
      inputTokens: 15,
      outputTokens: 12,
      responseTime: "180ms",
      ttft: "20ms",
      rawRequest: {
        prompt: "Write a haiku about artificial intelligence",
        max_tokens: 50,
        temperature: 0.7,
      },
      rawResponse: {
        text: "Silicon dreams wake\nNeural networks pulse with thought\nHuman-made conscience",
      },
      system: "You are a helpful AI assistant specialized in writing haikus.",
      inputMessages: [
        {
          role: "user",
          content: "Write a haiku about artificial intelligence",
        },
      ],
      output: {
        role: "assistant",
        content:
          "Silicon dreams wake\nNeural networks pulse with thought\nHuman-made conscience",
      },
    },
    {
      id: "5678-1234-9876-5432",
      model: "my_model",
      modelProvider: "my_provider",
      timestamp: "2025-01-06T13:16:21.800Z",
      inputTokens: 20,
      outputTokens: 14,
      responseTime: "220ms",
      ttft: "25ms",
      rawRequest: {
        prompt: "Refine the haiku to focus more on the future of AI",
        max_tokens: 50,
        temperature: 0.5,
      },
      rawResponse: {
        text: "Quantum minds evolve\nSingularity approaches\nHumanity's next leap",
      },
      system: "You are a helpful AI assistant specialized in writing haikus.",
      inputMessages: [
        {
          role: "user",
          content: "Refine the haiku to focus more on the future of AI",
        },
      ],
      output: {
        role: "assistant",
        content:
          "Quantum minds evolve\nSingularity approaches\nHumanity's next leap",
      },
    },
  ],
};

export default function InferenceDetail() {
  return (
    <div className="container mx-auto space-y-6 p-4">
      <h2 className="mb-4 text-2xl font-semibold">
        Inference{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">
          {inferenceData.id}
        </code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>

      <Card>
        <CardHeader>
          <CardTitle>Basic Information</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4">
            <div>
              <dt className="font-semibold">Function</dt>
              <dd>
                <Code>{inferenceData.function}</Code>
              </dd>
            </div>
            <div>
              <dt className="font-semibold">Variant</dt>
              <dd className="flex items-center gap-2">
                <Code>{inferenceData.variant}</Code>
                <Badge variant="destructive">{inferenceData.variantType}</Badge>
              </dd>
            </div>
            <div>
              <dt className="font-semibold">Episode ID</dt>
              <dd>
                <Code>{inferenceData.episodeId}</Code>
              </dd>
            </div>
            <div>
              <dt className="font-semibold">Timestamp</dt>
              <dd>{new Date(inferenceData.timestamp).toLocaleString()}</dd>
            </div>
            <div>
              <dt className="font-semibold">Processing Time</dt>
              <dd>{inferenceData.processingTime}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(inferenceData.input, null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(inferenceData.output, null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feedback</CardTitle>
        </CardHeader>
        <CardContent>
          {/* @VIRAJ: Use ~same as episode detail view. */}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead>Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell>
                  <Code>accuracy</Code>
                </TableCell>
                <TableCell>37%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>
                  <Code>valid_json</Code>
                </TableCell>
                <TableCell>True</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>
                  <Code>comment</Code>
                </TableCell>
                <TableCell>This is bad</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Inference Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(inferenceData.inferenceParameters, null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tool Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted p-4">
            <code className="text-sm">
              {JSON.stringify(inferenceData.toolParameters, null, 2)}
            </code>
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tags</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Key</TableHead>
                <TableHead>Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(inferenceData.tags).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell>
                    <Code>{key}</Code>
                  </TableCell>
                  <TableCell>
                    <Code>{value}</Code>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Inferences</CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {inferenceData.modelInferences.map((inference, index) => (
              <AccordionItem value={`item-${index}`} key={inference.id}>
                <AccordionTrigger>
                  <Code>{inference.id}</Code>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <dt className="font-semibold">Model</dt>
                        <dd>
                          <Code>{inference.model}</Code>
                        </dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Model Provider</dt>
                        <dd>
                          <Code>{inference.modelProvider}</Code>
                        </dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Input Tokens</dt>
                        <dd>{inference.inputTokens}</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Output Tokens</dt>
                        <dd>{inference.outputTokens}</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Response Time</dt>
                        <dd>{inference.responseTime}</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">TTFT</dt>
                        <dd>{inference.ttft}</dd>
                      </div>
                      <div>
                        <dt className="font-semibold">Timestamp</dt>
                        <dd>
                          {new Date(inference.timestamp).toLocaleString()}
                        </dd>
                      </div>
                    </div>

                    <Card>
                      <CardHeader>
                        <CardTitle>System Message</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <pre className="overflow-x-auto rounded-md bg-muted p-4">
                          <code className="text-sm">
                            {JSON.stringify(inference.system, null, 2)}
                          </code>
                        </pre>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Input Messages</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <pre className="overflow-x-auto rounded-md bg-muted p-4">
                          <code className="text-sm">
                            {JSON.stringify(inference.inputMessages, null, 2)}
                          </code>
                        </pre>
                      </CardContent>
                    </Card>

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
                            {JSON.stringify(inference.rawRequest, null, 2)}
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
                            {JSON.stringify(inference.rawResponse, null, 2)}
                          </code>
                        </pre>
                      </CardContent>
                    </Card>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>
    </div>
  );
}
