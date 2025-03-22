import type {
  JsonInferenceOutput,
  ContentBlockOutput,
} from "~/utils/clickhouse/common";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  SnippetGroup,
  SnippetMessage,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { CodeMessage, TextMessage } from "~/components/layout/SnippetContent";

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
        <CodeMessage
          key={index}
          label="Text"
          content={block.text}
          showLineNumbers={true}
        />
      );
    case "tool_call":
      return (
        <CodeMessage
          key={index}
          label={`Tool: ${block.name}`}
          content={JSON.stringify(block.arguments, null, 2)}
          showLineNumbers={true}
        />
      );
  }
}

export function OutputContent({ output }: OutputProps) {
  if (isJsonInferenceOutput(output)) {
    const tabs: SnippetTab[] = [
      {
        id: "raw",
        label: "Raw Output",
        content: <CodeMessage content={output.raw} showLineNumbers={true} />,
      },
    ];

    if (output.parsed) {
      tabs.unshift({
        id: "parsed",
        label: "Parsed Output",
        content: (
          <CodeMessage
            content={JSON.stringify(output.parsed, null, 2)}
            showLineNumbers={true}
          />
        ),
      });
    }

    return <SnippetTabs tabs={tabs} />;
  }

  return (
    <SnippetGroup>
      <SnippetMessage>
        {output.map((block, index) => renderContentBlock(block, index))}
      </SnippetMessage>
    </SnippetGroup>
  );
}

export default function Output({ output }: OutputProps) {
  return (
    <SnippetLayout className="w-full">
      <SnippetContent>
        <OutputContent output={output} />
      </SnippetContent>
    </SnippetLayout>
  );
}
