import {
  CodeEditor,
  formatOptionalJson,
  useMemoizedFormat,
} from "~/components/ui/code-editor";
import { useVariantSelection, usePlaygroundFunctionAtom } from "../state";
import clsx from "clsx";
import AnimatedCollapsible from "../ui/AnimatedCollapsible";
import { cn } from "~/utils/common";
import type { FunctionConfig } from "tensorzero-node";
import { useAtom } from "jotai";
import { Checkbox } from "~/components/ui/checkbox";
import { LayoutGroup, motion } from "motion/react";
import { playgroundGrid } from "../layout";

export default function FunctionPanel({
  functionName,
  functionConfig,
  className,
}: {
  functionName: string;
  functionConfig: FunctionConfig;
  className?: string;
}) {
  const { variants, selectedVariants, addVariant, removeVariant } =
    useVariantSelection(functionName, functionConfig);

  const functionAtom = usePlaygroundFunctionAtom(functionName);
  const [functionLayout, setFunctionLayout] = useAtom(functionAtom);
  const {
    showSystemPrompt,
    showAssistantPrompt,
    showUserPrompt,
    showOutputSchema,
  } = functionLayout;

  const systemSchema = useMemoizedFormat(
    functionConfig.system_schema?.value,
    formatOptionalJson,
  );
  const assistantSchema = useMemoizedFormat(
    functionConfig.assistant_schema?.value,
    formatOptionalJson,
  );
  const userSchema = useMemoizedFormat(
    functionConfig.user_schema?.value,
    formatOptionalJson,
  );
  const outputSchema = useMemoizedFormat(
    functionConfig.type === "json"
      ? functionConfig.output_schema?.value
      : undefined,
    formatOptionalJson,
  );

  return (
    <motion.div
      layout
      className={cn(
        "row-span-full grid grid-cols-1 grid-rows-subgrid gap-y-0.5",
        "relative z-10 rounded-2xl bg-white p-4 shadow-md",
        "h-full min-h-0 overflow-scroll",
        className,
      )}
    >
      <LayoutGroup>
        {/* TODO Move function selection here? */}
        {/* <header className={playgroundGrid({ row: "header" })}></header> */}

        <motion.section
          key="variant-selection"
          layout="position"
          className={clsx(
            "flex-1 space-y-1 overflow-y-auto pb-2",
            playgroundGrid({ row: "metadata" }),
          )}
        >
          <h3>Variants</h3>

          {variants.map(([variantName]) => {
            // TODO Store `selected` on object/map instead - still need an array of selected variants for reordering later
            const isChecked = selectedVariants.some(
              ([name]) => name === variantName,
            );
            return (
              <label
                key={variantName}
                className="flex cursor-pointer items-center space-x-3 hover:text-gray-900"
              >
                <Checkbox
                  checked={isChecked}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      addVariant(variantName);
                    } else {
                      removeVariant(variantName);
                    }
                  }}
                  aria-label={`Select ${variantName}`}
                />
                <span className="flex-1 font-mono text-sm text-gray-700">
                  {variantName}
                </span>
              </label>
            );
          })}
        </motion.section>

        {(systemSchema || userSchema || assistantSchema || outputSchema) && (
          <>
            <motion.section
              key="prompt-section-header"
              layout="position"
              className={playgroundGrid({ row: "promptSectionHeader" })}
            >
              <h3>Schemas</h3>
            </motion.section>

            {systemSchema && (
              <motion.section
                key="system-schema"
                layout="position"
                className={playgroundGrid({ row: "system" })}
              >
                <AnimatedCollapsible
                  label="System"
                  isOpen={showSystemPrompt}
                  onOpenChange={(showSystemPrompt) =>
                    setFunctionLayout((prev) => ({ ...prev, showSystemPrompt }))
                  }
                >
                  <CodeEditor
                    readOnly
                    allowedLanguages={["json"]}
                    value={systemSchema}
                    className="bg-white"
                  />
                </AnimatedCollapsible>
              </motion.section>
            )}

            {assistantSchema && (
              <motion.section
                key="assistant-schema"
                layout="position"
                className={playgroundGrid({ row: "assistant" })}
              >
                <AnimatedCollapsible
                  label="Assistant"
                  isOpen={showAssistantPrompt}
                  onOpenChange={(showAssistantPrompt) =>
                    setFunctionLayout((prev) => ({
                      ...prev,
                      showAssistantPrompt,
                    }))
                  }
                >
                  <CodeEditor
                    readOnly
                    allowedLanguages={["json"]}
                    value={assistantSchema}
                    className="bg-white"
                  />
                </AnimatedCollapsible>
              </motion.section>
            )}

            {userSchema && (
              <motion.section
                key="user-schema"
                layout="position"
                className={playgroundGrid({ row: "user" })}
              >
                <AnimatedCollapsible
                  label="User"
                  isOpen={showUserPrompt}
                  onOpenChange={(showUserPrompt) =>
                    setFunctionLayout((prev) => ({ ...prev, showUserPrompt }))
                  }
                >
                  <CodeEditor
                    readOnly
                    allowedLanguages={["json"]}
                    value={userSchema}
                    className="bg-white"
                  />
                </AnimatedCollapsible>
              </motion.section>
            )}

            {outputSchema && (
              <motion.section
                key="output-schema"
                layout="position"
                className={playgroundGrid({ row: "outputSchema" })}
              >
                <AnimatedCollapsible
                  isOpen={showOutputSchema}
                  onOpenChange={(showOutputSchema) =>
                    setFunctionLayout((prev) => ({ ...prev, showOutputSchema }))
                  }
                  label="Output"
                >
                  <CodeEditor
                    readOnly
                    allowedLanguages={["json"]}
                    value={outputSchema}
                    className="bg-white"
                  />
                </AnimatedCollapsible>
              </motion.section>
            )}
          </>
        )}

        {/* TODO Add tool config here, too! */}
      </LayoutGroup>
    </motion.div>
  );
}
