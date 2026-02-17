import { Textarea } from "~/components/ui/textarea";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";
import { InlineMarkdown } from "./InlineMarkdown";

export function FreeResponseStep({
  question,
  text,
  onTextChange,
}: {
  question: Extract<EventPayloadUserQuestion, { type: "free_response" }>;
  text: string;
  onTextChange: (text: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <span className="text-fg-primary text-sm font-medium">
        <InlineMarkdown text={question.question} />
      </span>
      <Textarea
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        placeholder="Type your response..."
        className="bg-bg-secondary min-h-[80px] resize-none text-sm"
        rows={3}
        autoFocus
      />
    </div>
  );
}
