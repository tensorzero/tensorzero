import { atom, type WritableAtom } from "jotai";

// Helper function to immutably set a value at a nested path
function deepSetPath<T>(
  obj: Record<string, unknown>,
  path: string[],
  value: T,
): Record<string, unknown> {
  if (path.length === 0) return value as unknown as Record<string, unknown>;

  const [head, ...tail] = path;
  const current = obj?.[head];

  if (tail.length === 0) {
    return { ...obj, [head]: value };
  }

  return {
    ...obj,
    [head]: deepSetPath(
      (current ?? {}) as Record<string, unknown>,
      tail,
      value,
    ),
  };
}

// Utility for deriving atoms from a nested path
export function createRecordAtom<T, R>(
  baseAtom: WritableAtom<R, [R], void>,
  path: string[],
  defaultValue: T,
): WritableAtom<T, [React.SetStateAction<T>], void> {
  return atom(
    (get) => {
      let current: unknown = get(baseAtom);
      for (const key of path) {
        current = (current as Record<string, unknown>)?.[key];
        if (current === undefined) break;
      }
      return (current as T) ?? defaultValue;
    },
    (get, set, update) => {
      const base = get(baseAtom);
      const currentValue =
        path.reduce(
          (acc: unknown, key) => (acc as Record<string, unknown>)?.[key],
          base as unknown,
        ) ?? defaultValue;
      const newValue =
        typeof update === "function"
          ? (update as (prev: T) => T)(currentValue as T)
          : update;

      const updated = deepSetPath(
        base as Record<string, unknown>,
        path,
        newValue,
      );
      set(baseAtom, updated as R);
    },
  );
}
