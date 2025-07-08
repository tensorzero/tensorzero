import type { VariantConfig } from "tensorzero-node";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
  type SnippetTab,
  SnippetMessage,
} from "~/components/layout/SnippetLayout";
import { TextMessage } from "~/components/layout/SnippetContent";

interface VariantTemplateProps {
  variantConfig: VariantConfig;
}

// Create a template tab with appropriate indicator based on content
function createTemplateTab(
  id: string,
  label: string,
  content?: string,
  emptyMessage?: string,
): SnippetTab & { emptyMessage?: string } {
  return {
    id,
    label,
    indicator: content ? "content" : "empty",
    emptyMessage,
  };
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
    const templates = {
      system:
        variantConfig.system_template?.contents ??
        variantConfig.system_template?.path ??
        "",
      user:
        variantConfig.user_template?.contents ??
        variantConfig.user_template?.path ??
        "",
      assistant:
        variantConfig.assistant_template?.contents ??
        variantConfig.assistant_template?.path ??
        "",
    };

    const tabs = [
      createTemplateTab(
        "system",
        "System Template",
        templates.system,
        "No system template defined.",
      ),
      createTemplateTab(
        "user",
        "User Template",
        templates.user,
        "No user template defined.",
      ),
      createTemplateTab(
        "assistant",
        "Assistant Template",
        templates.assistant,
        "No assistant template defined.",
      ),
    ];

    // Find the first tab with content, or default to "system"
    const defaultTab =
      tabs.find((tab) => tab.indicator === "content")?.id || "system";

    return (
      <SnippetLayout>
        <SnippetTabs tabs={tabs} defaultTab={defaultTab}>
          {(activeTab) => {
            const tab = tabs.find((tab) => tab.id === activeTab);
            const template = templates[activeTab as keyof typeof templates];

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
      createTemplateTab(
        "system_instructions",
        "System Instructions",
        content,
        "No system instructions defined.",
      ),
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
