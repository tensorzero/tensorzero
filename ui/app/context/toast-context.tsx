import * as React from "react";
import type { ToastActionElement, ToastProps } from "~/components/ui/toast";

export interface ToastState {
  effects: (() => void | (() => void))[];
  toasts: Map<string, ToasterToast>;
}

export type Toast = Omit<ToasterToast, "id">;

export type ToastAction =
  | { type: "ADD_TOAST"; toast: ToasterToast }
  | {
      type: "UPDATE_TOAST";
      id: ToasterToast["id"];
      toast: Partial<Toast>;
    }
  | {
      type: "QUEUE_DISMISS_TOAST";
      id?: ToasterToast["id"];
      addToRemoveQueue: (id: string) => (() => void) | void;
    }
  | { type: "REMOVE_TOAST"; id?: ToasterToast["id"] };

export interface ToastEmitResult {
  id: ToasterToast["id"];
  dismiss: (args?: { immediate?: boolean }) => void;
  update: (props: ToasterToast) => void;
}

export type ToastActionProps = Toast & { log?: boolean | string };

export interface Toaster {
  info: (props: ToastActionProps) => ToastEmitResult;
  success: (props: Omit<ToastActionProps, "variant">) => ToastEmitResult;
  error: (props: Omit<ToastActionProps, "variant">) => ToastEmitResult;
  warn: (props: Omit<ToastActionProps, "variant">) => ToastEmitResult;
  dismiss: (id: string) => void;
  dismissAll: () => void;
  update: (id: string, props: Partial<Toast>) => void;
}

export const ToastActionType = {
  ADD_TOAST: "ADD_TOAST",
  UPDATE_TOAST: "UPDATE_TOAST",
  QUEUE_DISMISS_TOAST: "QUEUE_DISMISS_TOAST",
  REMOVE_TOAST: "REMOVE_TOAST",
} as const satisfies { [K in ToastAction["type"]]: K };

const ToastContext = React.createContext<{
  toasts: ToastState["toasts"];
  toaster: Toaster;
} | null>(null);
ToastContext.displayName = "ToastContext";

export { ToastContext };

export type ToasterToast = ToastProps & {
  id: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
  action?: ToastActionElement;
};
