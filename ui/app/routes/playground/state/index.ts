import { useAtom, useAtomValue } from "jotai";
import { atomWithStorage } from "jotai/utils";
import { useMemo, useCallback } from "react";
import type {
  VariantConfig,
  FunctionConfig,
  VariantInfo,
} from "tensorzero-node";
import { createRecordAtom } from "./helpers";
import {
  DEFAULT_FUNCTION_STATE,
  DEFAULT_PLAYGROUND_STATE,
  DEFAULT_VARIANT_STATE,
  PlaygroundState,
} from "./model";
import { createStorage } from "./localstorage";

export const playgroundAtom = atomWithStorage<PlaygroundState>(
  "tensorzero:playground",
  DEFAULT_PLAYGROUND_STATE,
  createStorage(PlaygroundState),
);

export const usePlaygroundFunctionAtom = (functionName: string) =>
  useMemo(
    () =>
      createRecordAtom(
        playgroundAtom,
        ["functions", functionName],
        DEFAULT_FUNCTION_STATE,
      ),
    [functionName],
  );

export const usePlaygroundVariantAtom = (
  functionName: string,
  variantName: string,
) =>
  useMemo(
    () =>
      createRecordAtom(
        playgroundAtom,
        ["functions", functionName, "variants", variantName],
        DEFAULT_VARIANT_STATE,
      ),
    [functionName, variantName],
  );

/** Return edited or original variant templates */
export const useVariantTemplates = (
  functionName: string,
  variantName: string,
  originalVariant: VariantConfig,
) => {
  const variantAtom = usePlaygroundVariantAtom(functionName, variantName);
  const variantState = useAtomValue(variantAtom);

  return useMemo(() => {
    const originalSystemTemplate =
      originalVariant.type === "chat_completion"
        ? originalVariant.system_template?.contents
        : undefined;
    const originalAssistantTemplate =
      originalVariant.type === "chat_completion"
        ? originalVariant.assistant_template?.contents
        : undefined;
    const originalUserTemplate =
      originalVariant.type === "chat_completion"
        ? originalVariant.user_template?.contents
        : undefined;

    return {
      systemTemplate: variantState.systemTemplate ?? originalSystemTemplate,
      assistantTemplate:
        variantState.assistantTemplate ?? originalAssistantTemplate,
      userTemplate: variantState.userTemplate ?? originalUserTemplate,
      // Also return whether each template has been edited
      isSystemTemplateEdited:
        variantState.systemTemplate !== originalSystemTemplate,
      isAssistantTemplateEdited:
        variantState.assistantTemplate !== originalAssistantTemplate,
      isUserTemplateEdited: variantState.userTemplate !== originalUserTemplate,
    };
  }, [variantState, originalVariant]);
};

export const useVariantSelection = (
  functionName: string,
  functionConfig: FunctionConfig,
  maxVariants: number = 3,
) => {
  const functionAtom = usePlaygroundFunctionAtom(functionName);
  const [{ selectedVariants }, setFunctionState] = useAtom(functionAtom);

  const variantEntries = useMemo(
    () =>
      Object.entries(functionConfig.variants)
        .filter((variant): variant is [string, VariantInfo] => !!variant[1])
        .map(([name, variant]) => [name, variant.inner] as const),
    [functionConfig.variants],
  );

  const selectedVariantEntries = useMemo(
    () =>
      selectedVariants
        .map((name) => {
          const variant = functionConfig.variants[name];
          if (!variant) return null;
          return [name, variant.inner] as const;
        })
        .filter((entry): entry is [string, VariantConfig] => entry !== null),
    [selectedVariants, functionConfig.variants],
  );

  const setVariantOrder = useCallback(
    (newSelectedVariants: string[]) => {
      setFunctionState((state) => ({
        ...state,
        selectedVariants: newSelectedVariants,
      }));
    },
    [setFunctionState],
  );

  const addVariant = useCallback(
    (variantName: string) => {
      if (
        !selectedVariants.includes(variantName) &&
        selectedVariants.length < maxVariants
      ) {
        setVariantOrder([...selectedVariants, variantName]);
      }
    },
    [selectedVariants, setVariantOrder, maxVariants],
  );

  const removeVariant = useCallback(
    (variantName: string) => {
      setVariantOrder(selectedVariants.filter((v) => v !== variantName));
    },
    [selectedVariants, setVariantOrder],
  );

  return useMemo(
    () => ({
      variants: variantEntries,
      selectedVariants: selectedVariantEntries,
      setVariantOrder,
      addVariant,
      removeVariant,
    }),
    [
      variantEntries,
      selectedVariantEntries,
      setVariantOrder,
      addVariant,
      removeVariant,
    ],
  );
};
