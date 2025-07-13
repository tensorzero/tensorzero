import type { SyncStorage } from "jotai/vanilla/utils/atomWithStorage";
import { PlaygroundState } from "./model";
import type { ZodType } from "zod";

// Don't subscribe to `storage` events given multiple open playground tabs:
// the layout in one tab shouldn't suddenly change because a user adjusts/edits something in another.
// Last-write-wins is probably best.
export const createStorage = (validator: ZodType) =>
  ({
    getItem: (key, initialValue) => {
      try {
        const item = localStorage.getItem(key);
        if (!item) {
          return initialValue;
        }

        const parsed = JSON.parse(item);
        return validator.parse(parsed);
      } catch {
        return initialValue;
      }
    },
    setItem: (key, newValue) => {
      localStorage.setItem(key, JSON.stringify(newValue));
    },
    removeItem: (key) => {
      localStorage.removeItem(key);
    },
  }) satisfies SyncStorage<PlaygroundState>;
