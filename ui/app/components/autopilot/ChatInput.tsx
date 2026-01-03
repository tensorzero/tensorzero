import { SendHorizontal, Loader2 } from "lucide-react";
import {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  type KeyboardEvent,
} from "react";
import { Textarea } from "~/components/ui/textarea";
import { cn } from "~/utils/common";

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
  className?: string;
  isNewSession?: boolean;
};

export function ChatInput({
  sessionId,
  onMessageSent,
  onMessageFailed,
  disabled = false,
  className,
  isNewSession = false,
}: ChatInputProps) {
  const [text, setText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const previousUserMessageEventIdRef = useRef<string | undefined>(undefined);

  // Reset idempotency cursor when session changes
  useEffect(() => {
    previousUserMessageEventIdRef.current = undefined;
  }, [sessionId]);

  // Sample a random placeholder for new sessions, default for existing sessions
  const placeholder = useMemo(
    () =>
      isNewSession
        ? PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)]
        : "Type a message...",
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

  const handleSend = useCallback(async () => {
    const trimmedText = text.trim();
    if (!trimmedText || isSending) return;

    setIsSending(true);

    try {
      const response = await fetch(
        `/api/autopilot/sessions/${sessionId}/events/message`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: trimmedText,
            previous_user_message_event_id:
              previousUserMessageEventIdRef.current,
          }),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to send message");
      }

      const data = (await response.json()) as {
        event_id: string;
        session_id: string;
      };

      // Store for idempotency on next message
      previousUserMessageEventIdRef.current = data.event_id;

      setText("");
      onMessageSent?.(data, trimmedText);
    } catch (error) {
      const err = error instanceof Error ? error : new Error("Unknown error");
      onMessageFailed?.(err);
    } finally {
      setIsSending(false);
    }
  }, [text, isSending, sessionId, onMessageSent, onMessageFailed]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const canSend = text.trim().length > 0 && !isSending && !disabled;

  return (
    <div className={cn("flex items-end gap-2", className)}>
      <Textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled || isSending}
        className="resize-none overflow-y-auto"
        style={{ minHeight: MIN_HEIGHT, maxHeight: MAX_HEIGHT }}
        rows={1}
      />
      <button
        type="button"
        onClick={handleSend}
        disabled={!canSend}
        className={cn(
          "flex h-[44px] w-[44px] shrink-0 cursor-pointer items-center justify-center rounded-md",
          "bg-fg-primary text-bg-primary hover:bg-fg-secondary",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "transition-colors",
        )}
        aria-label="Send message"
      >
        {isSending ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : (
          <SendHorizontal className="h-5 w-5" />
        )}
      </button>
    </div>
  );
}
