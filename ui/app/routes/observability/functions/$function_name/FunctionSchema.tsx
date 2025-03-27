import type { FunctionConfig, JSONSchema } from "~/utils/config/function";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { CodeMessage } from "~/components/layout/SnippetContent";

interface FunctionSchemaProps {
  functionConfig: FunctionConfig;
}

function formatSchema(schema?: JSONSchema) {
  if (!schema) {
    return "No schema defined";
  }
  return JSON.stringify(schema, null, 2);
}

// Create a schema tab with appropriate indicator based on schema content
function createSchemaTab(
  id: string,
  label: string,
  schema?: JSONSchema,
): SnippetTab {
  return {
    id,
    label,
    indicator: schema ? "content" : "empty",
  };
}

export default function FunctionSchema({
  functionConfig,
}: FunctionSchemaProps) {
  const schemas = {
    system: functionConfig.system_schema?.content,
    user: functionConfig.user_schema?.content,
    assistant: functionConfig.assistant_schema?.content,
    ...(functionConfig.type === "json"
      ? { output: functionConfig.output_schema?.content }
      : {}),
  };

  const tabs: SnippetTab[] = [
    createSchemaTab("system", "System Schema", schemas.system),
    createSchemaTab("user", "User Schema", schemas.user),
    createSchemaTab("assistant", "Assistant Schema", schemas.assistant),
    ...(functionConfig.type === "json"
      ? [createSchemaTab("output", "Output Schema", schemas.output)]
      : []),
  ];

  // Find the first tab with content, or default to "system"
  const defaultTab =
    tabs.find((tab) => tab.indicator === "content")?.id || "system";

  return (
    <SnippetLayout>
      <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
        {(activeTab) => (
          <SnippetContent>
            <CodeMessage
              content={formatSchema(schemas[activeTab as keyof typeof schemas])}
              showLineNumbers={true}
            />
          </SnippetContent>
        )}
      </SnippetTabs>
    </SnippetLayout>
  );
}
