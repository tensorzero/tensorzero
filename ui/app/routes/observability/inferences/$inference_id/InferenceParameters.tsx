import {
  SnippetLayout,
  SnippetContent,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { CodeMessage } from "~/components/layout/SnippetContent";

interface ParameterCardProps {
  parameters: string;
}

export function ParameterCard({ parameters }: ParameterCardProps) {
  return (
    <SnippetLayout>
      <SnippetContent>
        <SnippetMessage>
          <CodeMessage
            content={parameters}
            showLineNumbers={true}
            emptyMessage="No parameters defined"
          />
        </SnippetMessage>
      </SnippetContent>
    </SnippetLayout>
  );
}
