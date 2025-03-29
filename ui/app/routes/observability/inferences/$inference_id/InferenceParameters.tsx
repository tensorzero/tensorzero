import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeMessage } from "~/components/layout/SnippetContent";

interface ParameterCardProps {
  parameters: Record<string, unknown>;
}

export function ParameterCard({ parameters }: ParameterCardProps) {
  return (
    <SnippetLayout>
      <SnippetContent>
        <CodeMessage
          content={JSON.stringify(parameters, null, 2)}
          showLineNumbers={true}
          emptyMessage="No parameters defined"
        />
      </SnippetContent>
    </SnippetLayout>
  );
}
