import { atom } from "jotai";

export const showSystemPromptAtom = atom(false);
export const showAssistantPromptAtom = atom(false);
export const showUserPromptAtom = atom(false);

// TODO Figure out a way to track *per function* state so it's preserved if the user changes it
export const playgroundLayoutAtom = atom<{
  functions: {
    [functionName: string]: {
      /** Order of the selected variants for this function */
      selectedVariants: string[];

      showSystemPrompt: boolean;
      showAssistantPrompt: boolean;
      showUserPrompt: boolean;
      showOutputSchema: boolean;

      // TODO I think this should all be saved in local storage
    };
  };
}>({ functions: {} });

/**
 * TODO What other functionality do I want here...?
 */

// TODO For each visible datapoint - what panels are visible? Input, output, metrics?

/** Selected datapoints in Playground */
export const datapointsAtom = atom<string[]>([
  "0197b21f-5e78-7a92-931a-20ad51930336", // TODO fix
]);
