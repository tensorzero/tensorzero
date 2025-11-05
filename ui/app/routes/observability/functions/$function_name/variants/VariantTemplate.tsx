import type { VariantConfig } from "~/types/tensorzero";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { TextMessage } from "~/components/layout/SnippetContent";
import { LegacyStructuredPromptBadge } from "~/components/ui/LegacyStructuredPromptBadge";

interface VariantTemplateProps {
  variantConfig: VariantConfig;
}

export default function VariantTemplate({
  variantConfig,
}: VariantTemplateProps) {
  // Only render if we have templates to show
  if (
    variantConfig.type !== "chat_completion" &&
    variantConfig.type !== "dicl"
  ) {
    return null;
  }

  if (variantConfig.type === "chat_completion") {
    // Dynamically get all template names from the config
    const templateEntries = Object.entries(variantConfig.templates);

    // If no templates exist, show an empty state
    if (templateEntries.length === 0) {
      return (
        <SnippetLayout>
          <SnippetContent maxHeight={240}>
            <SnippetMessage>
              <TextMessage emptyMessage="No templates defined." />
            </SnippetMessage>
          </SnippetContent>
        </SnippetLayout>
      );
    }

    // Build templates object with all available templates
    const templates: Record<string, string> = {};
    for (const [name, templateData] of Object.entries(
      variantConfig.templates,
    )) {
      if (!templateData) continue;
      templates[name] = templateData.template.contents;
    }

    // Create tabs for each template
    const tabs = Object.entries(variantConfig.templates).map(
      ([name, templateData]) => {
        const isLegacy = templateData?.legacy_definition === true;
        return {
          id: name,
          label: (
            <div className="flex items-center gap-2">
              <span>{name}</span>
              {isLegacy && (
                <LegacyStructuredPromptBadge name={name} type="template" />
              )}
            </div>
          ),
          emptyMessage: "The template is empty.",
        };
      },
    );

    return (
      <SnippetLayout>
        <SnippetTabs tabs={tabs} defaultTab={tabs[0]?.id}>
          {(activeTab) => {
            const tab = tabs.find((tab) => tab.id === activeTab);
            const template = templates[activeTab];

            return (
              <SnippetContent maxHeight={240}>
                <SnippetMessage>
                  <TextMessage
                    content={template}
                    emptyMessage={tab?.emptyMessage}
                  />
                </SnippetMessage>
              </SnippetContent>
            );
          }}
        </SnippetTabs>
      </SnippetLayout>
    );
  }

  if (variantConfig.type === "dicl") {
    const content = variantConfig.system_instructions;

    const tabs = [
      {
        id: "system_instructions",
        label: "System Instructions",
        emptyMessage: "No system instructions defined.",
      },
    ];

    return (
      <SnippetLayout>
        <SnippetTabs tabs={tabs} defaultTab="system_instructions">
          {() => (
            <SnippetContent maxHeight={240}>
              <SnippetMessage>
                <TextMessage
                  content={content}
                  emptyMessage="No system instructions defined."
                />
              </SnippetMessage>
            </SnippetContent>
          )}
        </SnippetTabs>
      </SnippetLayout>
    );
  }

  return null;
}
