import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";
import { PendingQuestionCard } from "./PendingQuestionCard";
import { CompletedQuestionCard } from "./CompletedQuestionCard";

// ── Fixtures ──────────────────────────────────────────────────────────

const singleMcPayload: EventPayloadUserQuestions = {
  questions: [
    {
      id: "q1",
      header: "Auth method",
      question: "Which authentication method should we use for the API?",
      type: "multiple_choice",
      options: [
        {
          id: "jwt",
          label: "JWT tokens",
          description:
            "Stateless tokens signed with a secret key. Good for distributed systems.",
        },
        {
          id: "session",
          label: "Session cookies",
          description:
            "Server-side sessions with cookie-based authentication. Simpler but requires sticky sessions.",
        },
        {
          id: "oauth",
          label: "OAuth 2.0",
          description:
            "Delegated authorization with third-party providers like Google or GitHub.",
        },
      ],
      multi_select: false,
    },
  ],
};

const freeResponsePayload: EventPayloadUserQuestions = {
  questions: [
    {
      id: "q1",
      header: "Feedback",
      question:
        "What would you like to improve about the current implementation?",
      type: "free_response",
    },
  ],
};

const multiQuestionPayload: EventPayloadUserQuestions = {
  questions: [
    {
      id: "q1",
      header: "Library",
      question: "Which date formatting library should we use?",
      type: "multiple_choice",
      options: [
        {
          id: "datefns",
          label: "date-fns",
          description: "Lightweight, tree-shakeable date utility library.",
        },
        {
          id: "dayjs",
          label: "Day.js",
          description:
            "Tiny alternative to Moment.js with similar API. 2KB gzipped.",
        },
        {
          id: "luxon",
          label: "Luxon",
          description:
            "Modern DateTime library by the Moment.js team. Immutable and chainable.",
        },
      ],
      multi_select: false,
    },
    {
      id: "q2",
      header: "Features",
      question: "Which features do you want to enable?",
      type: "multiple_choice",
      options: [
        {
          id: "i18n",
          label: "Internationalization",
          description: "Support for multiple languages and locales.",
        },
        {
          id: "tz",
          label: "Timezone support",
          description: "Handle dates across different timezones correctly.",
        },
        {
          id: "relative",
          label: "Relative time",
          description: 'Display dates as "2 hours ago" or "in 3 days".',
        },
      ],
      multi_select: true,
    },
    {
      id: "q3",
      header: "Rationale",
      question:
        "Any additional context or constraints we should know about for this decision?",
      type: "free_response",
    },
  ],
};

const answeredResponses: Record<string, UserQuestionAnswer> = {
  q1: { type: "multiple_choice", selected: ["jwt"] },
};

const multiAnsweredResponses: Record<string, UserQuestionAnswer> = {
  q1: { type: "multiple_choice", selected: ["datefns"] },
  q2: { type: "multiple_choice", selected: ["i18n", "tz"] },
  q3: {
    type: "free_response",
    text: "We need timezone support because our users are global.",
  },
};

// ── Meta ──────────────────────────────────────────────────────────────

const meta = {
  title: "Autopilot/QuestionCards",
  component: PendingQuestionCard,
  render: (args) => (
    <div className="w-[500px] p-4">
      <PendingQuestionCard {...args} />
    </div>
  ),
} satisfies Meta<typeof PendingQuestionCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SingleMultipleChoice: Story = {
  args: {
    eventId: "evt-001",
    payload: singleMcPayload,
    isLoading: false,
    onSubmit: () => {},
    onSkip: () => {},
  },
};

const multiSelectPayload: EventPayloadUserQuestions = {
  questions: [
    {
      id: "q1",
      header: "Features",
      question: "Which features do you want to enable?",
      type: "multiple_choice",
      options: [
        {
          id: "i18n",
          label: "Internationalization",
          description: "Support for multiple languages and locales.",
        },
        {
          id: "tz",
          label: "Timezone support",
          description: "Handle dates across different timezones correctly.",
        },
        {
          id: "relative",
          label: "Relative time",
          description: 'Display dates as "2 hours ago" or "in 3 days".',
        },
      ],
      multi_select: true,
    },
  ],
};

export const SingleMultiSelect: Story = {
  args: {
    eventId: "evt-001b",
    payload: multiSelectPayload,
    isLoading: false,
    onSubmit: () => {},
    onSkip: () => {},
  },
};

export const SingleFreeResponse: Story = {
  args: {
    eventId: "evt-002",
    payload: freeResponsePayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MultiQuestion: Story = {
  args: {
    eventId: "evt-003",
    payload: multiQuestionPayload,
    isLoading: false,
    onSubmit: () => {},
    onSkip: () => {},
  },
};

export const Loading: Story = {
  args: {
    eventId: "evt-004",
    payload: singleMcPayload,
    isLoading: true,
    onSubmit: () => {},
  },
};

// Interactive story that logs submissions
function InteractivePending() {
  const [lastSubmission, setLastSubmission] = useState<Record<
    string,
    UserQuestionAnswer
  > | null>(null);

  return (
    <div className="flex w-[500px] flex-col gap-4 p-4">
      <PendingQuestionCard
        eventId="evt-interactive"
        payload={multiQuestionPayload}
        isLoading={false}
        onSubmit={(_id, responses) => setLastSubmission(responses)}
        onSkip={() => setLastSubmission(null)}
      />
      {lastSubmission && (
        <pre className="bg-bg-secondary text-fg-primary rounded-md border p-3 text-xs">
          {JSON.stringify(lastSubmission, null, 2)}
        </pre>
      )}
    </div>
  );
}

// Dummy args satisfy meta type — render overrides component entirely
const dummyArgs = {
  eventId: "",
  payload: singleMcPayload,
  isLoading: false,
  onSubmit: () => {},
};

export const Interactive: Story = {
  args: dummyArgs,
  render: () => <InteractivePending />,
};

// ── CompletedQuestionCard ─────────────────────────────────────────────

export const AnsweredSingle: Story = {
  args: dummyArgs,
  render: () => (
    <div className="w-[500px] p-4">
      <CompletedQuestionCard
        payload={singleMcPayload}
        responses={answeredResponses}
        eventId="evt-answered-1"
        timestamp={new Date().toISOString()}
      />
    </div>
  ),
};

export const AnsweredMulti: Story = {
  args: dummyArgs,
  render: () => (
    <div className="w-[500px] p-4">
      <CompletedQuestionCard
        payload={multiQuestionPayload}
        responses={multiAnsweredResponses}
        eventId="evt-answered-2"
        timestamp={new Date().toISOString()}
      />
    </div>
  ),
};

export const Skipped: Story = {
  args: dummyArgs,
  render: () => (
    <div className="w-[500px] p-4">
      <CompletedQuestionCard
        payload={multiQuestionPayload}
        eventId="evt-skipped-1"
        timestamp={new Date().toISOString()}
      />
    </div>
  ),
};
