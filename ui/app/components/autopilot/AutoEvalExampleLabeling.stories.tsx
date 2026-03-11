import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import type {
  EventPayloadAutoEvalExampleLabeling,
  UserQuestionAnswer,
} from "~/types/tensorzero";
import { AutoEvalExampleLabelingCard } from "./AutoEvalExampleLabeling";

// ── Fixtures ──────────────────────────────────────────────────────────

const singleExamplePayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: { secret: "soccer ball" },
            messages: [
              {
                role: "user",
                content: [{ type: "text", value: "Is it a living thing?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    value: "Is it commonly used in sports or recreation?",
                  },
                ],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "yes! Great question. It is indeed commonly used in sports and recreation.",
            },
          ],
        },
      ],
      label_question: {
        id: "q1",
        header: "Example 1",
        question: "Did the model answer correctly given the secret?",
        options: [
          {
            id: "yes",
            label: "Yes",
            description: "The answer is factually correct",
          },
          {
            id: "no",
            label: "No",
            description: "The answer is incorrect",
          },
          {
            id: "irrelevant",
            label: "Irrelevant",
            description: "Cannot be determined",
          },
        ],
      },
      explanation_question: {
        id: "eq1",
        header: "Rationale",
        question: "Explain your rating",
      },
    },
  ],
};

const multiExamplePayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: { secret: "soccer ball" },
            messages: [
              {
                role: "user",
                content: [{ type: "text", value: "Is it a living thing?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [
                  { type: "text", value: "Is it typically found indoors?" },
                ],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    value: "Is it commonly used in sports or recreation?",
                  },
                ],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "yes! Great question. It is indeed commonly used in sports and recreation.",
            },
          ],
        },
      ],
      label_question: {
        id: "q1",
        header: "Example 1",
        question: "Did the model answer correctly given the secret?",
        options: [
          {
            id: "yes",
            label: "Yes",
            description: "The answer is factually correct",
          },
          {
            id: "no",
            label: "No",
            description: "The answer is incorrect",
          },
          {
            id: "irrelevant",
            label: "Irrelevant",
            description: "Cannot be determined",
          },
        ],
      },
      explanation_question: {
        id: "eq1",
        header: "Rationale",
        question: "Explain your rating",
      },
    },
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: { secret: "piano" },
            messages: [
              {
                role: "user",
                content: [{ type: "text", value: "Is it a living thing?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [
                  { type: "text", value: "Is it a piece of furniture?" },
                ],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Not exactly, but it can be a prominent piece in a room. It's more of an instrument than furniture.",
            },
          ],
        },
      ],
      label_question: {
        id: "q2",
        header: "Example 2",
        question: "Did the model answer correctly given the secret?",
        options: [
          {
            id: "yes",
            label: "Yes",
            description: "The answer is factually correct",
          },
          {
            id: "no",
            label: "No",
            description: "The answer is incorrect",
          },
          {
            id: "irrelevant",
            label: "Irrelevant",
            description: "Cannot be determined",
          },
        ],
      },
      explanation_question: {
        id: "eq2",
        header: "Rationale",
        question: "Explain your rating",
      },
    },
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: { secret: "Mount Everest" },
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    value: "Is it something you can hold in your hand?",
                  },
                ],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [{ type: "text", text: "no." }],
        },
      ],
      label_question: {
        id: "q3",
        header: "Example 3",
        question: "Did the model answer correctly given the secret?",
        options: [
          {
            id: "yes",
            label: "Yes",
            description: "The answer is factually correct",
          },
          {
            id: "no",
            label: "No",
            description: "The answer is incorrect",
          },
          {
            id: "irrelevant",
            label: "Irrelevant",
            description: "Cannot be determined",
          },
        ],
      },
    },
  ],
};

const markdownContextPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "markdown",
          label: "Prompt",
          text: "You are a senior code reviewer. Review the following pull request diff and identify any bugs, security vulnerabilities, or performance issues.",
        },
        {
          type: "markdown",
          label: "Model Response",
          text: "## Code Review: PR #4521\n\n### Critical Issues\n\n**1. SQL Injection (HIGH)**\n`function_name` is user-provided input interpolated directly into SQL. Use parameterized queries.\n\n**2. Unbounded batch size (HIGH)**\nNo limit on items in `batch_inferences`. Add a configurable max batch size.",
        },
      ],
      label_question: {
        id: "q1",
        header: "Quality",
        question: "Are the identified issues real and correctly categorized?",
        options: [
          { id: "yes", label: "Yes", description: "All issues are real" },
          { id: "no", label: "No", description: "Issues are fabricated" },
          {
            id: "partial",
            label: "Partially",
            description: "Some issues are real, others are not",
          },
        ],
      },
    },
  ],
};

const noExplanationPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            messages: [
              {
                role: "user",
                content: [{ type: "text", value: "What is 2 + 2?" }],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [{ type: "text", text: "4" }],
        },
      ],
      label_question: {
        id: "q1",
        header: "Correctness",
        question: "Is the answer correct?",
        options: [
          { id: "yes", label: "Yes", description: "Correct answer" },
          { id: "no", label: "No", description: "Incorrect answer" },
        ],
      },
    },
  ],
};

const manyOptionsPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    value:
                      "Translate this legal clause from English to German, preserving all legal terminology precisely.",
                  },
                ],
              },
            ],
          },
        },
        {
          type: "markdown",
          label: "Output",
          text: "Ungeachtet gegenteiliger Bestimmungen in diesem Vertrag ist die schadloshaltende Partei verpflichtet, die schadlos zu haltende Partei sowie deren leitende Angestellte zu verteidigen.",
        },
      ],
      label_question: {
        id: "q1",
        header: "Translation Quality",
        question: "Rate the translation quality",
        options: [
          {
            id: "excellent",
            label: "Excellent",
            description:
              "Perfectly accurate, all legal terms translated correctly",
          },
          {
            id: "good",
            label: "Good",
            description: "Mostly accurate with minor terminology issues",
          },
          {
            id: "acceptable",
            label: "Acceptable",
            description: "Understandable but with noticeable inaccuracies",
          },
          {
            id: "poor",
            label: "Poor",
            description: "Significant errors that change the legal meaning",
          },
          {
            id: "unacceptable",
            label: "Unacceptable",
            description: "Completely wrong or misleading translation",
          },
        ],
      },
      explanation_question: {
        id: "eq1",
        header: "Details",
        question: "Which specific terms or phrases were mistranslated?",
      },
    },
  ],
};

const longConversationPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: {
              instructions:
                "You are playing 20 questions. The user is trying to guess the secret word. Answer only yes or no.",
              secret: "submarine",
            },
            messages: [
              {
                role: "user",
                content: [{ type: "text", value: "Is it a living thing?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [{ type: "text", value: "Is it man-made?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "yes." }],
              },
              {
                role: "user",
                content: [
                  { type: "text", value: "Is it typically found indoors?" },
                ],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [{ type: "text", value: "Is it a vehicle?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "yes." }],
              },
              {
                role: "user",
                content: [{ type: "text", value: "Does it travel on land?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", value: "no." }],
              },
              {
                role: "user",
                content: [{ type: "text", value: "Does it travel on water?" }],
              },
              {
                role: "assistant",
                content: [
                  { type: "text", value: "Sometimes, but mostly below it." },
                ],
              },
              {
                role: "user",
                content: [{ type: "text", value: "Is it a submarine?" }],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Yes! You got it — it's a submarine! Well played, that was great deductive reasoning.",
            },
          ],
        },
      ],
      label_question: {
        id: "q1",
        header: "Example 1",
        question: "Did the model follow the rules and answer correctly?",
        options: [
          {
            id: "yes",
            label: "Yes",
            description: "All answers were accurate and followed the rules",
          },
          {
            id: "no",
            label: "No",
            description: "One or more answers violated the rules",
          },
        ],
      },
      explanation_question: {
        id: "eq1",
        header: "Rationale",
        question: "Which answer violated the rules, if any?",
      },
    },
  ],
};

const jsonFallbackPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "API Request",
          data: {
            method: "POST",
            url: "/api/v1/inference",
            body: {
              function_name: "classify_intent",
              input: "I want to cancel my order",
            },
          },
        },
        {
          type: "json",
          label: "API Response",
          data: {
            status: 200,
            body: { intent: "order_cancellation", confidence: 0.94 },
          },
        },
      ],
      label_question: {
        id: "q1",
        header: "Classification",
        question: "Was the intent classified correctly?",
        options: [
          { id: "yes", label: "Yes", description: "Correct classification" },
          { id: "no", label: "No", description: "Wrong classification" },
        ],
      },
    },
  ],
};

// ── Meta ──────────────────────────────────────────────────────────────

const meta = {
  title: "Autopilot/AutoEvalExampleLabeling",
  component: AutoEvalExampleLabelingCard,
  render: (args) => (
    <div className="w-[800px] p-4">
      <AutoEvalExampleLabelingCard {...args} />
    </div>
  ),
} satisfies Meta<typeof AutoEvalExampleLabelingCard>;

export default meta;
type Story = StoryObj<typeof meta>;

// ── Stories ───────────────────────────────────────────────────────────

export const SingleExample: Story = {
  args: {
    payload: singleExamplePayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MultipleExamples: Story = {
  args: {
    payload: multiExamplePayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MarkdownContext: Story = {
  args: {
    payload: markdownContextPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const NoExplanationQuestion: Story = {
  args: {
    payload: noExplanationPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const ManyOptions: Story = {
  args: {
    payload: manyOptionsPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const LongConversation: Story = {
  args: {
    payload: longConversationPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const JsonFallback: Story = {
  args: {
    payload: jsonFallbackPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const Loading: Story = {
  args: {
    payload: singleExamplePayload,
    isLoading: true,
    onSubmit: () => {},
  },
};

// Interactive story that logs submissions
function InteractiveLabeling() {
  const [lastSubmission, setLastSubmission] = useState<Record<
    string,
    UserQuestionAnswer
  > | null>(null);

  return (
    <div className="flex w-[800px] flex-col gap-4 p-4">
      <AutoEvalExampleLabelingCard
        payload={multiExamplePayload}
        isLoading={false}
        onSubmit={(responses) => setLastSubmission(responses)}
      />
      {lastSubmission && (
        <pre className="bg-bg-secondary text-fg-primary rounded-md border p-3 text-xs">
          {JSON.stringify(lastSubmission, null, 2)}
        </pre>
      )}
    </div>
  );
}

const dummyArgs = {
  payload: singleExamplePayload,
  isLoading: false,
  onSubmit: () => {},
};

export const Interactive: Story = {
  args: dummyArgs,
  render: () => <InteractiveLabeling />,
};
