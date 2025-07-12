import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";

interface ParameterCardProps {
  parameters: string;
}

export function ParameterCard({ parameters }: ParameterCardProps) {
  return (
    <SnippetLayout>
      <SnippetContent>
        <div className="w-full">
          <CodeEditor allowedLanguages={["json"]} value={parameters} readOnly />
        </div>
      </SnippetContent>
    </SnippetLayout>
  );
}
