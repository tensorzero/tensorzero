import { Card, CardContent } from "~/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import type { VariantConfig, DiclConfig } from "~/utils/config/variant";

interface VariantTemplateProps {
  variantConfig: VariantConfig;
}

interface TemplateFieldProps {
  title: string;
  content: string;
}

function TemplateField({ title, content }: TemplateFieldProps) {
  if (!content) {
    return (
      <div className="col-span-2">
        <dt className="text-lg font-semibold">{title}</dt>
        <dd className="text-sm text-muted-foreground">None</dd>
      </div>
    );
  }

  return (
    <div className="col-span-2">
      <dt className="text-lg font-semibold">{title}</dt>
      <dd>
        <Collapsible>
          <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
            <ChevronDown className="h-4 w-4" />
            Show full template
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="max-h-[500px] overflow-auto rounded-md bg-muted p-4">
              <code className="text-sm">{content}</code>
            </pre>
          </CollapsibleContent>
        </Collapsible>
      </dd>
    </div>
  );
}

export default function VariantTemplate({
  variantConfig,
}: VariantTemplateProps) {
  // Only render if we have templates to show
  if (
    variantConfig.type !== "chat_completion" &&
    variantConfig.type !== "experimental_dynamic_in_context_learning"
  ) {
    return null;
  }

  return (
    <Card className="mb-4">
      <CardContent className="p-6">
        <dl className="grid grid-cols-2 gap-4">
          {/* Chat Completion Templates */}
          {variantConfig.type === "chat_completion" && (
            <>
              <TemplateField
                title="System Template"
                content={
                  variantConfig.system_template?.content ??
                  variantConfig.system_template?.path ??
                  ""
                }
              />
              <TemplateField
                title="User Template"
                content={
                  variantConfig.user_template?.content ??
                  variantConfig.user_template?.path ??
                  ""
                }
              />
              <TemplateField
                title="Assistant Template"
                content={
                  variantConfig.assistant_template?.content ??
                  variantConfig.assistant_template?.path ??
                  ""
                }
              />
            </>
          )}

          {/* Dynamic In-Context Learning Template */}
          {variantConfig.type ===
            "experimental_dynamic_in_context_learning" && (
            <TemplateField
              title="System Instructions"
              content={
                (variantConfig as DiclConfig).system_instructions?.content ??
                (variantConfig as DiclConfig).system_instructions?.path ??
                ""
              }
            />
          )}
        </dl>
      </CardContent>
    </Card>
  );
}
