import {
  useCallback,
  useEffect,
  useSyncExternalStore,
  type Dispatch,
  type SetStateAction,
} from "react";

type Value = string | boolean | object | null | undefined;

function dispatchStorageEvent(key: string, newValue: Value) {
  const _newValue =
    newValue !== null && newValue !== undefined
      ? JSON.stringify(newValue)
      : newValue;
  window.dispatchEvent(
    new StorageEvent("storage", {
      key,
      newValue: _newValue,
    }),
  );
}

const setLocalStorageItem = (key: string, value: Value) => {
  const stringifiedValue = JSON.stringify(value);
  window.localStorage.setItem(key, stringifiedValue);
  dispatchStorageEvent(key, stringifiedValue);
};

const removeLocalStorageItem = (key: string) => {
  window.localStorage.removeItem(key);
  dispatchStorageEvent(key, null);
};

const getLocalStorageItem = (key: string) => {
  return window.localStorage.getItem(key);
};

const useLocalStorageSubscribe = (callback: (ev: StorageEvent) => void) => {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
};

const getLocalStorageServerSnapshot = () => {
  return null;
};

/**
 * A hook to synchronize data with the browser's localStorage API,
 * for persistent client-side storage.
 * @returns [value, setValue], similar to useState()
 */
export function useLocalStorage<T extends Value>(
  key: string,
  initialValue?: T,
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const getSnapshot = () => getLocalStorageItem(key);

  const store = useSyncExternalStore(
    useLocalStorageSubscribe,
    getSnapshot,
    getLocalStorageServerSnapshot,
  );

  const setState = useCallback<Dispatch<SetStateAction<T>>>(
    (v) => {
      try {
        const nextState = typeof v === "function" ? v(JSON.parse(store!)) : v;

        if (nextState === undefined || nextState === null) {
          removeLocalStorageItem(key);
        } else {
          setLocalStorageItem(key, nextState);
        }
      } catch (e) {
        console.warn(e);
      }
    },
    [key, store],
  );

  useEffect(() => {
    if (
      getLocalStorageItem(key) === null &&
      typeof initialValue !== "undefined"
    ) {
      setLocalStorageItem(key, initialValue);
    }
  }, [key, initialValue]);

  return [store ? JSON.parse(store) : initialValue, setState];
}
