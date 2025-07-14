import type { FunctionConfig, JsonValue } from "tensorzero-node";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { CodeEditor } from "~/components/ui/code-editor";

interface FunctionSchemaProps {
  functionConfig: FunctionConfig;
}

// Create a schema tab with appropriate indicator based on schema content
function createSchemaTab(
  id: string,
  label: string,
  schema?: JsonValue,
  emptyMessage?: string,
): SnippetTab & { emptyMessage?: string } {
  return {
    id,
    label,
    indicator: schema ? "content" : "empty",
    emptyMessage,
  };
}

export default function FunctionSchema({
  functionConfig,
}: FunctionSchemaProps) {
  const schemas = {
    system: functionConfig.system_schema?.value,
    user: functionConfig.user_schema?.value,
    assistant: functionConfig.assistant_schema?.value,
    ...(functionConfig.type === "json"
      ? { output: functionConfig.output_schema?.value }
      : {}),
  };

  const tabs = [
    createSchemaTab(
      "system",
      "System Schema",
      schemas.system,
      "No system schema defined.",
    ),
    createSchemaTab(
      "user",
      "User Schema",
      schemas.user,
      "No user schema defined.",
    ),
    createSchemaTab(
      "assistant",
      "Assistant Schema",
      schemas.assistant,
      "No assistant schema defined.",
    ),
    ...(functionConfig.type === "json"
      ? [
          createSchemaTab(
            "output",
            "Output Schema",
            schemas.output,
            "No output schema defined.",
          ),
        ]
      : []),
  ];

  // Find the first tab with content, or default to "system"
  const defaultTab =
    tabs.find((tab) => tab.indicator === "content")?.id || "system";

  return (
    <SnippetLayout>
      <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
        {(activeTab) => {
          const tab = tabs.find((tab) => tab.id === activeTab);
          const schema = schemas[activeTab as keyof typeof schemas];
          const formattedContent = schema
            ? JSON.stringify(schema, null, 2)
            : undefined;

          return (
            <SnippetContent maxHeight={240}>
              {formattedContent ? (
                <CodeEditor
                  allowedLanguages={["json"]}
                  value={formattedContent}
                  readOnly
                />
              ) : (
                <EmptyMessage message={tab?.emptyMessage} />
              )}
            </SnippetContent>
          );
        }}
      </SnippetTabs>
    </SnippetLayout>
  );
}
