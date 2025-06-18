import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { CodeMessage } from "~/components/layout/SnippetContent";
import { CodeBlock } from "~/components/ui/code-block";

interface ParameterCardProps {
  parameters: string;
  html?: string | null;
}

export function ParameterCard({ parameters, html }: ParameterCardProps) {
  return (
    <SnippetLayout>
      <SnippetContent>
        {html ? (
          <div className="w-full">
            <CodeBlock html={html} showLineNumbers={true} />
          </div>
        ) : (
          <SnippetMessage>
            <CodeMessage
              content={parameters}
              showLineNumbers={true}
              emptyMessage="No parameters defined"
            />
          </SnippetMessage>
        )}
      </SnippetContent>
    </SnippetLayout>
  );
}
