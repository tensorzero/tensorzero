import type {
  ChatInferenceOutputRenderingData,
  JsonInferenceOutputRenderingData,
} from "~/components/inference/NewOutput";
import {
  TextMessage,
  ToolCallMessage,
} from "~/components/layout/SnippetContent";
import {
  SnippetLayout,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import {
  CodeEditor,
  formatJson,
  useMemoizedFormat,
} from "~/components/ui/code-editor";

// TODO Replace `NewOutput` with this
function InferenceOutput({
  outputs,
}: {
  outputs: ChatInferenceOutputRenderingData | JsonInferenceOutputRenderingData;
}) {
  return (
    <SnippetLayout>
      {Array.isArray(outputs) ? (
        outputs.map((output, i) => (
          <SnippetMessage key={i}>
            {output.type === "text" ? (
              <TextMessage content={output.text} />
            ) : (
              <ToolCallMessage
                toolName={output.raw_name}
                toolArguments={output.raw_arguments}
                toolCallId={output.id}
              />
            )}
          </SnippetMessage>
        ))
      ) : (
        <StructuredOutput output={outputs.raw} />
      )}
    </SnippetLayout>
  );
}

function StructuredOutput({ output }: { output: string }) {
  const formattedJson = useMemoizedFormat(output, formatJson);
  return <CodeEditor allowedLanguages={["json"]} value={formattedJson} />;
}

export default InferenceOutput;
