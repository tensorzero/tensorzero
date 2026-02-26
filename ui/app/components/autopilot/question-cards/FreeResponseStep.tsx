import { Textarea } from "~/components/ui/textarea";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";

type FreeResponseStepProps = {
  question: Extract<EventPayloadUserQuestion, { type: "free_response" }>;
  text: string;
  onTextChange: (text: string) => void;
};

export function FreeResponseStep({
  question,
  text,
  onTextChange,
}: FreeResponseStepProps) {
  return (
    <div className="flex flex-col gap-3">
      <span className="text-fg-primary text-sm font-medium">
        {question.question}
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
