import type { FunctionConfig, SchemaData } from "tensorzero-node";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
} from "~/components/layout/SnippetLayout";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { CodeEditor } from "~/components/ui/code-editor";
import { LegacyStructuredPromptBadge } from "~/components/ui/LegacyStructuredPromptBadge";

interface FunctionSchemaProps {
  functionConfig: FunctionConfig;
}

export default function FunctionSchema({
  functionConfig,
}: FunctionSchemaProps) {
  // Build schemas object dynamically from all available schemas
  const schemas: SchemaData = {
    ...functionConfig.schemas,
    ...(functionConfig.type === "json"
      ? {
          output: {
            schema: { value: functionConfig.output_schema?.value },
            legacy_definition: false,
          },
        }
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
  const tabs = schemaEntries.map(([name, schemaData]) => {
    const isLegacy = schemaData?.legacy_definition ?? false;
    const isOutputSchema = name === "output";
    return {
      id: name,
      label: (
        <div className="flex items-center gap-2">
          <span>{name}</span>
          {isLegacy && !isOutputSchema && (
            <LegacyStructuredPromptBadge name={name} type="schema" />
          )}
        </div>
      ),
      emptyMessage: "No schema defined.",
    };
  });

  return (
    <SnippetLayout>
      <SnippetTabs tabs={tabs} defaultTab={tabs[0]?.id}>
        {(activeTab) => {
          const tab = tabs.find((tab) => tab.id === activeTab);
          const schema = schemas[activeTab];
          const formattedContent = schema?.schema?.value
            ? JSON.stringify(schema.schema.value, null, 2)
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
