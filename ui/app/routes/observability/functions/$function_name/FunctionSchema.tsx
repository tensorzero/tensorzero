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

// Create a schema tab
function createSchemaTab(
  id: string,
  label: string,
  emptyMessage?: string,
): SnippetTab & { emptyMessage?: string } {
  return {
    id,
    label,
    emptyMessage,
  };
}

export default function FunctionSchema({
  functionConfig,
}: FunctionSchemaProps) {
  // Build schemas object dynamically from all available schemas
  const schemas: Record<string, JsonValue | undefined> = {
    ...Object.fromEntries(
      Object.entries(functionConfig.schemas).map(([name, schemaData]) => [
        name,
        schemaData?.value,
      ]),
    ),
    ...(functionConfig.type === "json"
      ? { output: functionConfig.output_schema?.value }
      : {}),
  };

  // Get all schema entries
  const schemaEntries = Object.entries(schemas);

  // If no schemas exist, show an empty state
  if (schemaEntries.length === 0) {
    return (
      <SnippetLayout>
        <SnippetContent maxHeight={240}>
          <EmptyMessage message="No schemas defined." />
        </SnippetContent>
      </SnippetLayout>
    );
  }

  // Create tabs for each schema
  const tabs = schemaEntries.map(([name]) => {
    return createSchemaTab(name, name, "No schema defined.");
  });

  // Default to the first tab
  const defaultTab = tabs[0]?.id;

  return (
    <SnippetLayout>
      <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
        {(activeTab) => {
          const tab = tabs.find((tab) => tab.id === activeTab);
          const schema = schemas[activeTab];
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
