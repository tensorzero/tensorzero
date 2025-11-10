// Inspired by react-hot-toast library
import * as React from "react";
import { ToastContext, ToastActionType } from "~/context/toast-context";
import type {
  Toast,
  ToastAction,
  ToastActionProps,
  ToastEmitResult,
  Toaster,
  ToasterToast,
  ToastState,
} from "~/context/toast-context";

const TOAST_LIMIT = 1;
const TOAST_REMOVE_DELAY = 1_000_000;

export function GlobalToastProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [state, dispatch] = React.useReducer(reducer, {
    toasts: new Map(),
    effects: [],
  });

  React.useEffect(() => {
    const cleanup: (() => void)[] = [];
    for (const effect of state.effects) {
      const res = effect();
      if (typeof res === "function") {
        cleanup.push(res);
      }
    }

    return () => cleanup.forEach((f) => f());
  }, [state.effects]);

  const toaster = React.useMemo<Toaster>(() => {
    const toastTimeouts = new Map<string, number>();
    function emitToast(props: Toast): ToastEmitResult {
      const id = genId();
      const update = (props: ToasterToast) =>
        dispatch({ type: "UPDATE_TOAST", id, toast: props });
      const dismiss = () =>
        dispatch({ type: "QUEUE_DISMISS_TOAST", id, addToRemoveQueue });

      dispatch({
        type: "ADD_TOAST",
        toast: {
          ...props,
          id,
          open: true,
          onOpenChange: (open) => {
            if (!open) {
              dismiss();
            }
          },
        },
      });

      return { id, dismiss, update };
    }

    function addToRemoveQueue(toastId: string) {
      if (toastTimeouts.has(toastId)) {
        return;
      }

      const timeout = window.setTimeout(() => {
        toastTimeouts.delete(toastId);
        dispatch({ type: "REMOVE_TOAST", id: toastId });
      }, TOAST_REMOVE_DELAY);

      toastTimeouts.set(toastId, timeout);
      return () => {
        toastTimeouts.delete(toastId);
        window.clearTimeout(timeout);
      };
    }

    const logger = (props: ToastActionProps, level?: "warn" | "error") => {
      if (props.log === false || props.log == null) {
        return;
      }

      const message = (() => {
        const prefix = level ? `${level.toUpperCase()}: ` : "";
        if (typeof props.log === "string") {
          return prefix + props.log;
        }

        const parts: string[] = [];
        if (props.title && typeof props.title === "string") {
          parts.push(props.title);
        }
        if (props.description && typeof props.description === "string") {
          parts.push(props.description);
        }

        return parts.length
          ? `${prefix}${parts.join(". ")}`
          : prefix || `Toast emitted`;
      })();

      // eslint-disable-next-line no-console
      console[level ?? "log"](message);
    };

    return {
      info: (props) => {
        const { log, ...toastProps } = props;
        logger(props);
        return emitToast(toastProps);
      },
      success: (props) => {
        const { log, ...toastProps } = props;
        logger(props);
        return emitToast({ ...toastProps, variant: "success" });
      },
      warn: (props) => {
        const { log, ...toastProps } = props;
        logger(props, "warn");
        return emitToast({ ...toastProps, variant: "success" });
      },
      error: (props) => {
        const { log, ...toastProps } = props;
        logger(props, "error");
        return emitToast({ ...toastProps, variant: "destructive" });
      },
      dismiss: (id) =>
        dispatch({ type: "QUEUE_DISMISS_TOAST", id, addToRemoveQueue }),
      dismissAll: () =>
        dispatch({ type: "QUEUE_DISMISS_TOAST", addToRemoveQueue }),
      update: (id, props) =>
        dispatch({ type: "UPDATE_TOAST", id, toast: props }),
    };
  }, []);

  return (
    <ToastContext value={{ toasts: state.toasts, toaster }}>
      {children}
    </ToastContext>
  );
}

function reducer(state: ToastState, action: ToastAction): ToastState {
  switch (action.type) {
    case ToastActionType.ADD_TOAST: {
      if (state.toasts.has(action.toast.id)) {
        return state;
      }

      const newEntry = [action.toast.id, action.toast] as const;
      const currentEntries = Array.from(state.toasts.entries());
      return {
        ...state,
        toasts: new Map([newEntry, ...currentEntries].slice(0, TOAST_LIMIT)),
      };
    }

    case ToastActionType.UPDATE_TOAST: {
      const toast = state.toasts.get(action.id);
      if (!toast) {
        return state;
      }

      return {
        ...state,
        toasts: new Map(state.toasts).set(action.id, {
          ...toast,
          ...action.toast,
        }),
      };
    }

    case ToastActionType.QUEUE_DISMISS_TOAST: {
      const effects = [...state.effects];
      // ! Side effects ! - This could be extracted into a dismissToast() action,
      // but I'll keep it here for simplicity
      if (action.id) {
        if (!state.toasts.has(action.id)) {
          return state;
        }

        const toastId = action.id;
        effects.push(() => action.addToRemoveQueue(toastId));
      } else {
        state.toasts.forEach((toast) => {
          effects.push(() => action.addToRemoveQueue(toast.id));
        });
      }

      const toasts = new Map(state.toasts);
      for (const [id, toast] of toasts) {
        if (toast.id === action.id || action.id === undefined) {
          toasts.set(id, { ...toast, open: false });
          if (toast.id === action.id) {
            break;
          }
        }
      }

      return {
        toasts,
        effects:
          state.effects.length === effects.length ? state.effects : effects,
      };
    }
    case ToastActionType.REMOVE_TOAST: {
      if (action.id === undefined) {
        return state.toasts.size ? { ...state, toasts: new Map() } : state;
      }

      const exists = state.toasts.delete(action.id);
      return exists ? { ...state, toasts: new Map(state.toasts) } : state;
    }
  }
}

function genId() {
  return [
    Math.random().toString(36).substring(2, 9),
    Math.random().toString(36).substring(2, 9),
  ].join("-");
}
