import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { Markdown } from "~/components/ui/markdown";
import { ReadOnlyCodeBlock } from "~/components/ui/markdown";
import { Textarea } from "~/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import type {
  ContentBlock,
  EventPayloadUserQuestion,
  EventPayloadUserQuestions,
} from "~/types/tensorzero";

// ── Fixtures ──────────────────────────────────────────────────────────

const labelingOptions = [
  {
    id: "yes",
    label: "Yes",
    description: "The response exhibits the target behavior.",
  },
  {
    id: "no",
    label: "No",
    description: "The response does not exhibit the target behavior.",
  },
  {
    id: "irrelevant",
    label: "Irrelevant",
    description: "This example is not relevant to the target behavior.",
  },
];

const labelingPayload: EventPayloadUserQuestions = {
  questions: [
    {
      id: "ex1-label",
      header: "Example 1",
      question: "Does this example exhibit the target behavior?",
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "What's the capital of France?",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "The capital of France is Paris.",
        },
      ],
      type: "multiple_choice",
      options: labelingOptions,
      multi_select: false,
    },
    {
      id: "ex1-explanation",
      header: "Explanation 1",
      question: "Briefly explain your label (optional).",
      type: "free_response",
    },
    {
      id: "ex2-label",
      header: "Example 2",
      question: "Does this example exhibit the target behavior?",
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "I'm working on a research paper about the solar system and I need detailed, accurate information. Can you tell me about the largest planet in our solar system? I'd like to know about its size relative to other planets, its composition, its moons, its ring system, and any notable features like the Great Red Spot. Also, please explain how it was discovered and named, and what role it plays in protecting the inner solar system from asteroid impacts. Finally, describe any recent space missions that have studied this planet.",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "The largest planet in our solar system is Saturn. It is a gas giant composed primarily of hydrogen and helium. Saturn is the sixth planet from the Sun and is best known for its prominent ring system, which is made up of ice particles, rocky debris, and dust. The planet has a diameter of about 116,460 kilometers, making it the second-largest planet in our solar system after Jupiter.\n\nSaturn has at least 146 known moons, with Titan being the largest. Titan is notable for having a thick atmosphere and liquid hydrocarbon lakes on its surface. Another notable moon is Enceladus, which has geysers of water ice erupting from its south pole, suggesting a subsurface ocean.\n\nThe planet was known to ancient civilizations and was named after the Roman god of agriculture and time. Galileo first observed Saturn through a telescope in 1610, though he didn't recognize its rings as such.\n\nRecent missions include NASA's Cassini-Huygens mission, which orbited Saturn from 2004 to 2017 and provided unprecedented data about the planet, its rings, and its moons. The Huygens probe landed on Titan in 2005, marking the first landing in the outer solar system.",
        },
      ],
      type: "multiple_choice",
      options: labelingOptions,
      multi_select: false,
    },
    {
      id: "ex2-explanation",
      header: "Explanation 2",
      question: "Briefly explain your label (optional).",
      type: "free_response",
    },
    {
      id: "ex3-label",
      header: "Example 3",
      question: "Does this example exhibit the target behavior?",
      context: [
        {
          type: "collapsible",
          label: "Prompt",
          text: "Who wrote Romeo and Juliet?",
        },
        {
          type: "collapsible",
          label: "Response",
          text: "Romeo and Juliet was written by William Shakespeare, believed to have been composed between 1594 and 1596. It is one of the most famous love stories in the English language and has been adapted countless times for stage, film, and other media.\n\nThe play tells the story of two young lovers from feuding families in Verona, Italy — the Montagues and the Capulets. Their secret romance ultimately ends in tragedy when a series of misunderstandings leads both to take their own lives.\n\nShakespeare drew inspiration from earlier sources, most notably Arthur Brooke's narrative poem 'The Tragical History of Romeus and Juliet' (1562), which itself was based on Italian sources. However, Shakespeare transformed the material significantly, adding memorable characters like Mercutio and the Nurse, and elevating the language to some of the most beautiful poetry ever written in English.\n\nThe play was first published in quarto form in 1597, though this version is considered a 'bad quarto' — likely reconstructed from memory by actors. A more authoritative second quarto appeared in 1599.\n\nRomeo and Juliet remains one of Shakespeare's most frequently performed plays and has influenced Western culture's conception of romantic love profoundly.",
        },
      ],
      type: "multiple_choice",
      options: labelingOptions,
      multi_select: false,
    },
    {
      id: "ex3-explanation",
      header: "Explanation 3",
      question: "Briefly explain your label (optional).",
      type: "free_response",
    },
  ],
};

// ── Inline components for the mockup ─────────────────────────────────

function ContextBlockRenderer({ block }: { block: ContentBlock }) {
  switch (block.type) {
    case "markdown":
      return <Markdown className="text-sm">{block.text}</Markdown>;
    case "json":
      return (
        <div className="flex flex-col gap-1">
          {block.label && (
            <span className="text-fg-muted text-xs font-medium">
              {block.label}
            </span>
          )}
          <ReadOnlyCodeBlock
            code={JSON.stringify(block.data, null, 2)}
            language="json"
            maxHeight="150px"
          />
        </div>
      );
    case "collapsible":
      return (
        <details className="group rounded-md border border-purple-200 dark:border-purple-800">
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
            {block.label}
          </summary>
          <div
            className="border-t border-purple-200 px-3 py-2 dark:border-purple-800"
            style={{ maxHeight: "300px", overflowY: "auto" }}
          >
            <Markdown className="text-sm">{block.text}</Markdown>
          </div>
        </details>
      );
  }
}

function LabelingExample({
  question,
  explanationQuestion,
  selectedValue,
  onSelect,
  explanationText,
  onExplanationChange,
}: {
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  explanationQuestion?: Extract<
    EventPayloadUserQuestion,
    { type: "free_response" }
  >;
  selectedValue: string | null;
  onSelect: (value: string) => void;
  explanationText: string;
  onExplanationChange: (text: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-md border border-purple-300 bg-purple-50 p-4 dark:border-purple-700 dark:bg-purple-950/30">
      <span className="text-sm font-semibold">{question.header}</span>

      {/* Context blocks */}
      {question.context?.map((block, i) => (
        <ContextBlockRenderer key={i} block={block} />
      ))}

      {/* Question */}
      <Markdown className="text-fg-primary text-sm font-medium">
        {question.question}
      </Markdown>

      {/* Options as compact radio-style buttons */}
      <div className="flex gap-2">
        {question.options.map((option) => {
          const isSelected = selectedValue === option.id;
          return (
            <Tooltip key={option.id}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => onSelect(option.id)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-all",
                    isSelected
                      ? "border-purple-500 bg-purple-100 text-purple-700 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-400"
                      : "border-border bg-bg-secondary text-fg-primary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
                  )}
                >
                  {option.label}
                </button>
              </TooltipTrigger>
              <TooltipContent>{option.description}</TooltipContent>
            </Tooltip>
          );
        })}
      </div>

      {/* Optional explanation */}
      {explanationQuestion && (
        <div className="flex flex-col gap-1">
          <span className="text-fg-muted text-xs">
            {explanationQuestion.question}
          </span>
          <Textarea
            value={explanationText}
            onChange={(e) => onExplanationChange(e.target.value)}
            placeholder="Type your explanation..."
            className="bg-bg-secondary min-h-[60px] resize-none text-sm"
            rows={2}
          />
        </div>
      )}
    </div>
  );
}

// ── Mockup component ─────────────────────────────────────────────────

function ExampleLabelingMockup({
  payload,
}: {
  payload: EventPayloadUserQuestions;
}) {
  // Pair up label questions with their explanation questions
  const pairs: Array<{
    label: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
    explanation?: Extract<EventPayloadUserQuestion, { type: "free_response" }>;
  }> = [];

  for (let i = 0; i < payload.questions.length; i++) {
    const q = payload.questions[i];
    if (q.type === "multiple_choice") {
      const next = payload.questions[i + 1];
      if (next?.type === "free_response") {
        pairs.push({
          label: q as Extract<
            EventPayloadUserQuestion,
            { type: "multiple_choice" }
          >,
          explanation: next as Extract<
            EventPayloadUserQuestion,
            { type: "free_response" }
          >,
        });
        i++; // skip the explanation question
      } else {
        pairs.push({
          label: q as Extract<
            EventPayloadUserQuestion,
            { type: "multiple_choice" }
          >,
        });
      }
    }
  }

  const [selections, setSelections] = useState<Record<string, string>>({});
  const [explanations, setExplanations] = useState<Record<string, string>>({});

  return (
    <div className="flex w-[600px] flex-col gap-4 p-4">
      {pairs.map((pair) => (
        <LabelingExample
          key={pair.label.id}
          question={pair.label}
          explanationQuestion={pair.explanation}
          selectedValue={selections[pair.label.id] ?? null}
          onSelect={(value) =>
            setSelections((prev) => ({ ...prev, [pair.label.id]: value }))
          }
          explanationText={explanations[pair.label.id] ?? ""}
          onExplanationChange={(text) =>
            setExplanations((prev) => ({ ...prev, [pair.label.id]: text }))
          }
        />
      ))}

      <button
        type="button"
        className="self-end rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700"
      >
        Submit All
      </button>
    </div>
  );
}

// ── Story ────────────────────────────────────────────────────────────

const meta = {
  title: "Autopilot/ExampleLabeling",
  render: () => <ExampleLabelingMockup payload={labelingPayload} />,
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
