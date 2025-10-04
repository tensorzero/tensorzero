import type { FunctionConfig, JsonValue } from "tensorzero-node";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
} from "~/components/layout/SnippetLayout";
import { EmptyMessage } from "~/components/layout/SnippetContent";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface FunctionSchemaProps {
  functionConfig: FunctionConfig;
}

export default function FunctionSchema({
  functionConfig,
}: FunctionSchemaProps) {
  // Build schemas object dynamically from all available schemas
  const schemas: Record<
    string,
    { value: JsonValue | undefined; legacy_definition: boolean }
  > = {
    ...Object.fromEntries(
      Object.entries(functionConfig.schemas).map(([name, schemaData]) => [
        name,
        {
          value: schemaData?.schema?.value,
          legacy_definition: schemaData?.legacy_definition ?? false,
        },
      ]),
    ),
    ...(functionConfig.type === "json"
      ? {
          output: {
            value: functionConfig.output_schema?.value,
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
    const isLegacy = schemaData.legacy_definition;
    const isOutputSchema = name === "output";
    return {
      id: name,
      label: (
        <div className="flex items-center gap-2">
          <span>{name}</span>
          {isLegacy && !isOutputSchema && (
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge className="bg-yellow-600 px-1 py-0 text-[10px] text-white">
                    Legacy
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs p-2">
                  <div className="text-xs">
                    Please migrate from <code>{name}_schema</code> to{" "}
                    <code>schemas.{name}.path</code>.{" "}
                    <a
                      href="https://www.tensorzero.com/docs/gateway/create-a-prompt-template#migrate-from-legacy-prompt-templates"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:text-gray-300"
                    >
                      Read more
                    </a>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      ),
      emptyMessage: "No schema defined.",
    };
  });

  // Default to the first tab
  const defaultTab = tabs[0]?.id;

  return (
    <SnippetLayout>
      <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
        {(activeTab) => {
          const tab = tabs.find((tab) => tab.id === activeTab);
          const schema = schemas[activeTab];
          const formattedContent = schema.value
            ? JSON.stringify(schema.value, null, 2)
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
