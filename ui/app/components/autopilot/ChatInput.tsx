import { SendHorizontal, Loader2, StopCircle } from "lucide-react";
import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  type KeyboardEvent,
} from "react";
import { useFetcher } from "react-router";
import { Textarea } from "~/components/ui/textarea";
import { cn } from "~/utils/common";

type MessageResponse =
  | { event_id: string; session_id: string; error?: never }
  | { error: string; event_id?: never; session_id?: never };

const MIN_HEIGHT = 44;
const MAX_HEIGHT = MIN_HEIGHT * 3; // 3x initial height

const PLACEHOLDERS = [
  "Why did inference 00000000-0000-0000-0000-000000000000 go wrong?",
  "Let's optimize the evaluation my_evaluation.",
];

type ChatInputProps = {
  sessionId: string;
  onMessageSent?: (
    response: { event_id: string; session_id: string },
    text: string,
  ) => void;
  onMessageFailed?: (error: Error) => void;
  disabled?: boolean;
  submitDisabled?: boolean;
  className?: string;
  isNewSession?: boolean;
  isInterruptible?: boolean;
  isInterrupting?: boolean;
  onInterrupt?: () => void;
};

export function ChatInput({
  sessionId,
  onMessageSent,
  onMessageFailed,
  disabled = false,
  submitDisabled = false,
  className,
  isNewSession = false,
  isInterruptible = false,
  isInterrupting = false,
  onInterrupt,
}: ChatInputProps) {
  const [text, setText] = useState("");
  const fetcher = useFetcher<MessageResponse>();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const previousUserMessageEventIdRef = useRef<string | undefined>(undefined);
  const pendingTextRef = useRef<string>("");

  // Store callbacks in refs to avoid re-triggering the effect when they change
  const onMessageSentRef = useRef(onMessageSent);
  const onMessageFailedRef = useRef(onMessageFailed);
  onMessageSentRef.current = onMessageSent;
  onMessageFailedRef.current = onMessageFailed;

  const isSubmitting = fetcher.state === "submitting";

  // Reset idempotency cursor when session changes
  useEffect(() => {
    previousUserMessageEventIdRef.current = undefined;
  }, [sessionId]);

  // Sample a random placeholder for new sessions, default for existing sessions
  const placeholder = useMemo(
    () =>
      isNewSession
        ? PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)]
        : "Send a message...",
    [isNewSession],
  );

  // Auto-resize textarea based on content
  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    // Reset height to auto to get accurate scrollHeight
    textarea.style.height = "auto";
    // Clamp between min and max height
    const newHeight = Math.min(
      Math.max(textarea.scrollHeight, MIN_HEIGHT),
      MAX_HEIGHT,
    );
    textarea.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [text, adjustTextareaHeight]);

  // Debounced resize handler for window width changes
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>;
    const handleResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(adjustTextareaHeight, 100);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener("resize", handleResize);
    };
  }, [adjustTextareaHeight]);

  // Handle fetcher response
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data) {
      const data = fetcher.data;
      if ("error" in data) {
        onMessageFailedRef.current?.(new Error(data.error));
      } else {
        previousUserMessageEventIdRef.current = data.event_id;
        setText("");
        onMessageSentRef.current?.(data, pendingTextRef.current);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            textareaRef.current?.focus();
          });
        });
      }
    }
  }, [fetcher.state, fetcher.data]);

  const canSend =
    text.trim().length > 0 && !isSubmitting && !disabled && !submitDisabled;

  const handleSend = useCallback(() => {
    const trimmedText = text.trim();
    if (!trimmedText || isSubmitting || submitDisabled) return;

    pendingTextRef.current = trimmedText;

    fetcher.submit(
      {
        text: trimmedText,
        ...(previousUserMessageEventIdRef.current && {
          previous_user_message_event_id: previousUserMessageEventIdRef.current,
        }),
      },
      {
        method: "POST",
        action: `/api/autopilot/sessions/${sessionId}/events/message`,
        encType: "application/json",
      },
    );
  }, [text, isSubmitting, submitDisabled, sessionId, fetcher]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (canSend) {
          handleSend();
        }
      }
    },
    [canSend, handleSend],
  );

  return (
    <div className={cn("relative", className)}>
      <Textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled || isSubmitting}
        className={cn(
          "bg-bg-secondary resize-none overflow-y-auto",
          "rounded-md py-[11px] pr-14 pl-4 text-sm",
          "focus-visible:border-fg-muted focus-visible:ring-0",
        )}
        style={{ minHeight: MIN_HEIGHT, maxHeight: MAX_HEIGHT }}
        rows={1}
      />
      {submitDisabled && isInterruptible ? (
        <button
          type="button"
          onClick={onInterrupt}
          disabled={isInterrupting}
          className={cn(
            "absolute right-2 bottom-1",
            "flex h-9 w-9 cursor-pointer items-center justify-center rounded-md",
            "text-red-600 hover:text-red-700",
            "disabled:cursor-not-allowed disabled:opacity-50",
            "transition-colors",
          )}
          aria-label="Stop session"
        >
          {isInterrupting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <StopCircle className="h-4 w-4" />
          )}
        </button>
      ) : (
        <button
          type="button"
          onClick={handleSend}
          disabled={!canSend}
          className={cn(
            "absolute right-2 bottom-1",
            "flex h-9 w-9 items-center justify-center rounded-md",
            "transition-colors",
            canSend
              ? "text-fg-primary hover:text-fg-secondary cursor-pointer"
              : "text-fg-muted cursor-not-allowed opacity-50",
          )}
          aria-label="Send message"
        >
          {isSubmitting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <SendHorizontal className="h-4 w-4" />
          )}
        </button>
      )}
    </div>
  );
}
