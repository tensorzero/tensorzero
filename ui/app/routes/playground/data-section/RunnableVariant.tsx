import type { ChatCompletionConfig, VariantConfig } from "tensorzero-node";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import { useVariantTemplates } from "../state";
import { useCallback, useEffect, useMemo, useState } from "react";
import { throttle } from "~/utils/throttle";
import { get_template_env } from "~/utils/config/variant";
import { useRunVariantInference } from "../queries";
import clsx from "clsx";
import RunButton from "../ui/RunButton";
import AnimatedCollapsible from "../ui/AnimatedCollapsible";
import InputSnippet from "~/components/inference/InputSnippet";
import { HEADER_ROW_HEIGHT_CSS_VAR } from "../ui/ResizableQuadrant";
import { datapointGrid } from "./layout";
import FadeTransition from "../ui/FadeTransition";
import { Loader2 } from "lucide-react";
import type { JsExposedEnv } from "~/utils/minijinja/pkg/minijinja_bindings";
import InferenceOutput from "../ui/InferenceOutput";

export function RunnableVariant({
  functionName,
  variantName,
  variant,
  datapoint,
  showInput,
  showOutput,
  setShowInput,
  setShowOutput,
}: {
  functionName: string;
  datapoint: ParsedDatasetRow;
  variantName: string;
  variant: VariantConfig;
  showInput: boolean;
  setShowInput: (showInput: boolean) => void;
  showOutput: boolean;
  setShowOutput: (showOutput: boolean) => void;
}) {
  // Use edited templates if available, otherwise fall back to original config
  const { systemTemplate, assistantTemplate, userTemplate } =
    useVariantTemplates(functionName, variantName, variant);

  // MiniJinja WASM requires configuring the instance with the templates ahead of time, which is async.
  // Throttle so it's not on every keystroke when we have editing.
  const [templateEnv, setTemplateEnv] = useState<JsExposedEnv>();
  const updateTemplateEnv = useMemo(
    () =>
      throttle(
        (
          variant: VariantConfig,
          systemTemplate: string | undefined,
          assistantTemplate: string | undefined,
          userTemplate: string | undefined,
        ) => {
          if (variant.type !== "chat_completion") {
            return;
          }

          // Create a modified variant config with effective templates
          const effectiveVariant: ChatCompletionConfig = {
            ...variant,
            system_template: systemTemplate
              ? { contents: systemTemplate, path: "" }
              : variant.system_template,
            assistant_template: assistantTemplate
              ? { contents: assistantTemplate, path: "" }
              : variant.assistant_template,
            user_template: userTemplate
              ? { contents: userTemplate, path: "" }
              : variant.user_template,
          };

          get_template_env(effectiveVariant).then((env) => {
            setTemplateEnv(env);
          });
        },
        300,
      ),
    [],
  );

  useEffect(
    () =>
      updateTemplateEnv(
        variant,
        systemTemplate,
        assistantTemplate,
        userTemplate,
      ),
    [
      updateTemplateEnv,
      variant,
      systemTemplate,
      assistantTemplate,
      userTemplate,
    ],
  );

  const renderedSystem = useMemo(() => {
    try {
      if (templateEnv?.has_template("system")) {
        // TODO Validate against schema first
        return templateEnv.render("system", datapoint.input.system);
      }
    } catch (error) {
      // TODO Potentially expose an error to the user
      console.error("Minijinja failed to render system template:", error);
    }

    return datapoint.input.system;
  }, [datapoint.input.system, templateEnv]);

  const renderedMessages = useMemo(() => {
    return datapoint.input.messages.map((message) => ({
      ...message,
      content: message.content.map((block): typeof block => {
        if (block.type === "structured_text") {
          try {
            // TODO Validate against schema first
            if (templateEnv?.has_template(message.role)) {
              const renderedBlock = templateEnv.render(
                message.role,
                block.arguments,
              );

              return {
                type: "raw_text",
                value: renderedBlock,
              };
            }
          } catch {
            // TODO Potentially expose an error to the user
            console.error(
              "Minijinja failed to render template. Datapoint input might not match the message schemas",
            );
          }
        }

        return block;
      }),
    }));
  }, [datapoint.input, templateEnv]);

  const inferenceMutation = useRunVariantInference(functionName, variantName);
  const runVariant = useCallback(() => {
    setShowOutput(true);
    inferenceMutation.mutate({
      // Cannot pass rendered system message - inference API always expects arguments for this.
      // Fix when we add support for dynamic variants.
      system: datapoint.input.system,
      messages: renderedMessages,
    });
  }, [datapoint, renderedMessages, inferenceMutation, setShowOutput]);

  const output = !inferenceMutation.data
    ? undefined
    : "content" in inferenceMutation.data
      ? inferenceMutation.data.content
      : inferenceMutation.data.output;

  return (
    <>
      <header
        className={clsx(
          "sticky flex w-full flex-row justify-end",
          datapointGrid({ row: "header" }),
        )}
        style={{
          // TODO Fix this
          top: `calc(var(${HEADER_ROW_HEIGHT_CSS_VAR}) + 2rem)`,
        }}
      >
        <RunButton onRun={runVariant} isLoading={inferenceMutation.isPending} />
      </header>

      <AnimatedCollapsible
        isOpen={showInput}
        onOpenChange={setShowInput}
        label="Input"
        className={datapointGrid({ row: "input" })}
      >
        <InputSnippet
          system={renderedSystem}
          messages={renderedMessages}
          maxHeight="Content"
        />
      </AnimatedCollapsible>

      <AnimatedCollapsible
        isOpen={showOutput}
        onOpenChange={setShowOutput}
        label="Output"
        className={clsx(datapointGrid({ row: "output" }))}
      >
        {inferenceMutation.isPending ? (
          <FadeTransition
            stateKey="loading"
            className="flex items-center gap-2 py-2"
          >
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-muted-foreground text-sm italic">
              Running...
            </span>
          </FadeTransition>
        ) : output ? (
          <FadeTransition stateKey="output">
            <InferenceOutput outputs={output} />
          </FadeTransition>
        ) : (
          <FadeTransition
            stateKey="no-runs"
            className="text-muted-foreground text-sm italic"
          >
            No runs
          </FadeTransition>
        )}
      </AnimatedCollapsible>
    </>
  );
}
