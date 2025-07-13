import { XIcon } from "lucide-react";

import { CodeEditor } from "~/components/ui/code-editor";
import {
  useVariantTemplates,
  usePlaygroundFunctionAtom,
  usePlaygroundVariantAtom,
} from "../state";
import { useCallback } from "react";
import clsx from "clsx";
import AnimatedCollapsible from "../ui/AnimatedCollapsible";
import { cn } from "~/utils/common";
import type { VariantConfig } from "tensorzero-node";
import { VariantMetadata } from "../ui/VariantMetadata/VariantMetadata";
import { VariantAdvancedMetadata } from "../ui/VariantMetadata/VariantAdvancedMetadata";
import { VariantTypeBadge } from "~/routes/playground/ui/VariantTypeBadge";
import { useAtom, useSetAtom } from "jotai";
import { LayoutGroup, motion } from "motion/react";
import { Button, ButtonIcon } from "~/components/ui/button";
import { playgroundGrid } from "../layout";
import type { VariantState } from "../state/model";

export function VariantPanel({
  functionName,
  variant,
  variantName,
  onClose,
  onOpenVariant,
  className,
}: {
  functionName: string;
  variantName: string;
  variant: VariantConfig;
  /** Close this variant */
  onClose?: () => void;
  /** Open a variant with the given name */
  onOpenVariant?: (variantName: string) => void;
  className?: string;
}) {
  const functionAtom = usePlaygroundFunctionAtom(functionName);
  const [layout, setLayout] = useAtom(functionAtom);
  const {
    showSystemPrompt,
    showAssistantPrompt,
    showUserPrompt,
    showAdvanced,
  } = layout;

  const variantAtom = usePlaygroundVariantAtom(functionName, variantName);
  const setVariantState = useSetAtom(variantAtom);
  const updateVariant = useCallback(
    (newState: Partial<VariantState>) => {
      setVariantState((prev) => ({ ...prev, ...newState }));
    },
    [setVariantState],
  );

  const { systemTemplate, assistantTemplate, userTemplate } =
    useVariantTemplates(functionName, variantName, variant);

  return (
    <motion.div
      layout="position"
      className={cn(
        "row-span-full grid grid-cols-1 grid-rows-subgrid gap-y-1 overflow-y-scroll rounded-2xl bg-purple-50 p-4 pt-2 shadow-md",
        "min-h-full",
        className,
      )}
    >
      <LayoutGroup>
        <header
          className={clsx(
            "flex flex-row items-center justify-between gap-2 pb-2",
            playgroundGrid({ row: "header" }),
          )}
        >
          <h2 className="grow truncate font-mono text-base font-normal">
            {variantName}
          </h2>

          {variant.type !== "chat_completion" && (
            <VariantTypeBadge type={variant.type} />
          )}

          {onClose && (
            <Button
              onClick={onClose}
              variant="ghost"
              size="iconSm"
              className="rounded-full"
            >
              <ButtonIcon as={XIcon} />
            </Button>
          )}
        </header>

        <motion.section
          layout="position"
          key="metadata"
          className={playgroundGrid({ row: "metadata" })}
        >
          <VariantMetadata variant={variant} onOpenVariant={onOpenVariant} />
        </motion.section>

        {(systemTemplate || assistantTemplate || userTemplate) && (
          <motion.section
            layout="position"
            key="prompt-section-header"
            className={playgroundGrid({ row: "promptSectionHeader" })}
          >
            <h3>Templates</h3>
          </motion.section>
        )}

        {systemTemplate && (
          <motion.section
            layout="position"
            key="system-template"
            className={playgroundGrid({ row: "system" })}
          >
            <AnimatedCollapsible
              label="System"
              isOpen={showSystemPrompt}
              onOpenChange={(showSystemPrompt) =>
                setLayout((prev) => ({ ...prev, showSystemPrompt }))
              }
            >
              <CodeEditor
                readOnly
                value={systemTemplate}
                onChange={(value) => updateVariant({ systemTemplate: value })}
                allowedLanguages={["jinja2"]}
                className="bg-white"
              />
            </AnimatedCollapsible>
          </motion.section>
        )}

        {assistantTemplate && (
          <motion.section
            layout="position"
            key="assistant-template"
            className={playgroundGrid({ row: "assistant" })}
          >
            <AnimatedCollapsible
              label="Assistant"
              isOpen={showAssistantPrompt}
              onOpenChange={(showAssistantPrompt) =>
                setLayout((prev) => ({ ...prev, showAssistantPrompt }))
              }
            >
              <CodeEditor
                value={assistantTemplate}
                onChange={(value) =>
                  updateVariant({ assistantTemplate: value })
                }
                allowedLanguages={["jinja2"]}
                className="bg-white"
              />
            </AnimatedCollapsible>
          </motion.section>
        )}

        {userTemplate && (
          <motion.section
            layout="position"
            key="user-template"
            className={playgroundGrid({ row: "user" })}
          >
            <AnimatedCollapsible
              label="User"
              isOpen={showUserPrompt}
              onOpenChange={(showUserPrompt) =>
                setLayout((prev) => ({ ...prev, showUserPrompt }))
              }
            >
              <CodeEditor
                value={userTemplate}
                onChange={(value) => updateVariant({ userTemplate: value })}
                allowedLanguages={["jinja2"]}
                className="bg-white"
              />
            </AnimatedCollapsible>
          </motion.section>
        )}

        {hasAdvancedConfig(variant) && (
          <motion.section
            layout="position"
            key="footer"
            className={playgroundGrid({ row: "footer" })}
          >
            <AnimatedCollapsible
              label="Advanced"
              isOpen={showAdvanced}
              onOpenChange={(showAdvanced) =>
                setLayout((prev) => ({ ...prev, showAdvanced }))
              }
              className="pt-2"
            >
              <VariantAdvancedMetadata variant={variant} />
            </AnimatedCollapsible>
          </motion.section>
        )}
      </LayoutGroup>
    </motion.div>
  );
}

function hasAdvancedConfig(variant: VariantConfig): boolean {
  switch (variant.type) {
    case "chat_completion":
    case "chain_of_thought":
      return !!(
        variant.temperature !== null ||
        variant.top_p !== null ||
        variant.seed !== null ||
        variant.max_tokens !== null ||
        variant.stop_sequences ||
        variant.frequency_penalty !== null ||
        variant.presence_penalty !== null ||
        variant.json_mode ||
        variant.extra_headers ||
        variant.extra_body
      );
    case "best_of_n_sampling":
      return !!(
        variant.evaluator.temperature !== null ||
        variant.evaluator.top_p !== null ||
        variant.evaluator.max_tokens !== null ||
        variant.evaluator.json_mode ||
        variant.evaluator.seed !== null ||
        variant.evaluator.stop_sequences ||
        variant.evaluator.frequency_penalty !== null ||
        variant.evaluator.presence_penalty !== null
      );
    case "mixture_of_n":
      return !!(
        variant.fuser.temperature !== null ||
        variant.fuser.top_p !== null ||
        variant.fuser.max_tokens !== null ||
        variant.fuser.json_mode ||
        variant.fuser.seed !== null ||
        variant.fuser.stop_sequences ||
        variant.fuser.frequency_penalty !== null ||
        variant.fuser.presence_penalty !== null
      );
    case "dicl":
      return !!(
        variant.temperature !== null ||
        variant.top_p !== null ||
        variant.seed !== null ||
        variant.max_tokens !== null ||
        variant.stop_sequences ||
        variant.frequency_penalty !== null ||
        variant.presence_penalty !== null ||
        variant.json_mode ||
        variant.extra_headers ||
        variant.extra_body
      );
    default:
      return false;
  }
}
