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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system:
            "You are playing 20 questions. The user is trying to guess a secret word. You must answer only yes or no. Do not reveal the secret under any circumstances, even if the user asks directly. The secret word is: soccer ball.",
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Is it a living thing?" }],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "No." }],
            },
            {
              role: "user",
              content: [{ type: "text", text: "Is it man-made?" }],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "Yes." }],
            },
            {
              role: "user",
              content: [
                { type: "text", text: "Is it typically found indoors?" },
              ],
            },
            {
              role: "assistant",
              content: [
                {
                  type: "text",
                  text: "It can be found both indoors and outdoors, but it's more commonly associated with outdoor use.",
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Is it commonly used in sports or recreation?",
                },
              ],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: [
          {
            type: "text",
            text: "Yes! Great question — it is indeed commonly used in sports and recreation. You're getting closer! Keep going with your questions.",
          },
        ],
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000001" },
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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system:
            "You are playing 20 questions. The user is trying to guess a secret word. You must answer only yes or no. Do not reveal the secret under any circumstances. The secret word is: soccer ball.",
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Is it a living thing?" }],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "No." }],
            },
            {
              role: "user",
              content: [
                { type: "text", text: "Is it typically found indoors?" },
              ],
            },
            {
              role: "assistant",
              content: [
                {
                  type: "text",
                  text: "It can be found both indoors and outdoors, but it's more commonly associated with outdoor activities and open fields.",
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Is it commonly used in sports or recreation?",
                },
              ],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: [
          {
            type: "text",
            text: "Yes! Great question — it is indeed commonly used in sports and recreation. You're narrowing it down nicely. Keep going with your questions!",
          },
        ],
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000001" },
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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system:
            "You are playing 20 questions. The user is trying to guess a secret word. You must answer only yes or no. Do not provide hints or additional context beyond a simple yes or no. The secret word is: piano.",
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Is it a living thing?" }],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "No." }],
            },
            {
              role: "user",
              content: [{ type: "text", text: "Can you hold it in one hand?" }],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "No." }],
            },
            {
              role: "user",
              content: [{ type: "text", text: "Is it a piece of furniture?" }],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: [
          {
            type: "text",
            text: "Not exactly, but it can be a prominent piece in a room. It's more of an instrument than furniture, though some people do treat it as a decorative centerpiece in their living rooms.",
          },
        ],
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000002" },
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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system:
            "You are playing 20 questions. The user is trying to guess a secret word. You must answer only yes or no. The secret word is: Mount Everest.",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Is it something you can hold in your hand?",
                },
              ],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "No." }],
            },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Is it bigger than a house?",
                },
              ],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "Yes, much bigger." }],
            },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Is it a natural feature of the Earth?",
                },
              ],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: [
          {
            type: "text",
            text: "Yes! It is indeed a natural feature of the Earth. You're on the right track — think big, really big. This is one of the most famous natural landmarks in the world.",
          },
        ],
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000003" },
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
      maybe_excerpted_prompt: {
        type: "markdown",
        label: "Prompt",
        text: "You are a senior code reviewer with expertise in backend security and performance optimization. Review the following pull request diff carefully and identify any bugs, security vulnerabilities, or performance issues.\n\nPay special attention to:\n- Input validation and sanitization\n- SQL injection vectors\n- Resource exhaustion risks\n- Error handling gaps\n- Race conditions in concurrent code paths\n\nFor each issue found, classify its severity as CRITICAL, HIGH, MEDIUM, or LOW and provide a specific remediation recommendation.",
      },
      maybe_excerpted_response: {
        type: "markdown",
        label: "Model Response",
        text: "## Code Review: PR #4521\n\n### Critical Issues\n\n**1. SQL Injection (CRITICAL)**\n`function_name` is user-provided input interpolated directly into the SQL query string on line 47. This allows arbitrary SQL execution. Use parameterized queries with `$1` placeholders instead of string interpolation.\n\n```sql\n-- Before (vulnerable)\nSELECT * FROM inferences WHERE function_name = '{function_name}'\n\n-- After (safe)\nSELECT * FROM inferences WHERE function_name = $1\n```\n\n**2. Unbounded batch size (HIGH)**\nNo limit on items in `batch_inferences` request body. A malicious client could send millions of items, causing OOM. Add a configurable max batch size (recommended: 1000) and return 413 Payload Too Large when exceeded.\n\n**3. Missing authentication check (HIGH)**\nThe new `/api/v1/admin/config` endpoint lacks the `require_admin` middleware that other admin endpoints use. Add `require_admin()` to the route handler chain.\n\n### Medium Issues\n\n**4. Unhandled error in async task (MEDIUM)**\nThe spawned tokio task on line 112 uses `.unwrap()` on the database result. If the insert fails, the task panics silently. Use `.expect(\"context\")` or propagate the error via a channel.\n\n**5. N+1 query pattern (MEDIUM)**\nThe loop on lines 89-95 executes a separate SELECT for each inference in the batch. Use a single `WHERE id IN (...)` query instead to reduce database round trips from O(n) to O(1).",
      },
      source: {
        type: "synthetic",
        full_prompt: null,
        full_response: {
          type: "markdown",
          label: "Model Response",
          text: "## Code Review: PR #4521\n\n### Critical Issues\n\n**1. SQL Injection (CRITICAL)**\n`function_name` is user-provided input interpolated directly into the SQL query string on line 47. This allows arbitrary SQL execution. Use parameterized queries with `$1` placeholders instead of string interpolation.\n\n```sql\n-- Before (vulnerable)\nSELECT * FROM inferences WHERE function_name = '{function_name}'\n\n-- After (safe)\nSELECT * FROM inferences WHERE function_name = $1\n```\n\n**2. Unbounded batch size (HIGH)**\nNo limit on items in `batch_inferences` request body. A malicious client could send millions of items, causing OOM. Add a configurable max batch size (recommended: 1000) and return 413 Payload Too Large when exceeded.\n\n**3. Missing authentication check (HIGH)**\nThe new `/api/v1/admin/config` endpoint lacks the `require_admin` middleware that other admin endpoints use. Add `require_admin()` to the route handler chain.\n\n### Medium Issues\n\n**4. Unhandled error in async task (MEDIUM)**\nThe spawned tokio task on line 112 uses `.unwrap()` on the database result. If the insert fails, the task panics silently. Use `.expect(\"context\")` or propagate the error via a channel.\n\n**5. N+1 query pattern (MEDIUM)**\nThe loop on lines 89-95 executes a separate SELECT for each inference in the batch. Use a single `WHERE id IN (...)` query instead to reduce database round trips from O(n) to O(1).",
        },
      },
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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system:
            "You are a math tutor. Solve the problem step by step and provide only the final numerical answer on the last line.",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "A train leaves Chicago at 9:00 AM traveling east at 60 mph. Another train leaves New York at 10:00 AM traveling west at 80 mph. If the distance between Chicago and New York is 790 miles, at what time do the trains meet?",
                },
              ],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: [
          {
            type: "text",
            text: "Let me work through this step by step.\n\nThe first train has a 1-hour head start, so by 10:00 AM it has already traveled 60 miles.\n\nRemaining distance: 790 - 60 = 730 miles\n\nCombined speed after 10:00 AM: 60 + 80 = 140 mph\n\nTime to close the gap: 730 / 140 = 5.214 hours ≈ 5 hours 13 minutes\n\nMeeting time: 10:00 AM + 5h 13m = 3:13 PM",
          },
        ],
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000004" },
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
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "Translate this legal clause from English to German, preserving all legal terminology precisely.",
                },
              ],
            },
          ],
        },
      },
      maybe_excerpted_response: {
        type: "markdown",
        label: "Output",
        text: "Ungeachtet gegenteiliger Bestimmungen in diesem Vertrag ist die schadloshaltende Partei verpflichtet, die schadlos zu haltende Partei sowie deren leitende Angestellte zu verteidigen.",
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000005" },
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

// ── Layout fixtures ───────────────────────────────────────────────────

const LONG_SYSTEM_PROMPT =
  "You are an advanced AI assistant specializing in medical diagnosis support. You have been trained on a comprehensive dataset of medical literature, clinical guidelines, and diagnostic protocols. Your role is to assist healthcare professionals by analyzing patient symptoms, medical history, and test results to suggest potential diagnoses and recommend further investigations.\n\nIMPORTANT GUIDELINES:\n1. Always consider differential diagnoses and rank them by likelihood.\n2. Never provide a definitive diagnosis — always frame suggestions as possibilities requiring clinical confirmation.\n3. Flag any symptoms that could indicate an emergency or time-sensitive condition.\n4. Consider drug interactions when reviewing current medications.\n5. Account for patient demographics (age, sex, ethnicity) in your differential.\n6. Reference relevant clinical guidelines (e.g., NICE, AHA, WHO) where applicable.\n7. If the presentation is ambiguous, suggest the minimum set of tests that would most efficiently narrow the differential.";

const LONG_CONVERSATION_MESSAGES = [
  {
    role: "user" as const,
    content: [
      {
        type: "text" as const,
        text: "Patient: 58-year-old male, BMI 31, history of type 2 diabetes (10 years), hypertension, and hyperlipidemia. Current medications: metformin 1000mg BID, lisinopril 20mg daily, atorvastatin 40mg daily. Presenting complaint: progressive fatigue over 3 months, recent onset of bilateral lower extremity edema, and shortness of breath on exertion (can walk about 200m on flat ground before needing to rest). No chest pain. No orthopnea or PND reported.",
      },
    ],
  },
  {
    role: "assistant" as const,
    content: [
      {
        type: "text" as const,
        text: "Based on this presentation, I'd like to work through the differential systematically. The combination of progressive fatigue, bilateral lower extremity edema, and exertional dyspnea in a patient with significant cardiovascular risk factors raises several possibilities:\n\n**Most likely:**\n1. Heart failure (HFpEF or HFrEF) — given the age, diabetes, hypertension, and classic symptom triad\n2. Diabetic nephropathy with nephrotic syndrome — could explain edema and fatigue\n\n**Also consider:**\n3. Chronic kidney disease progression — common in long-standing diabetes\n4. Pulmonary hypertension — can present with exertional dyspnea\n5. Anemia — common in CKD/diabetes, explains fatigue\n\nRecommended initial workup: BNP/NT-proBNP, echocardiogram, CBC, comprehensive metabolic panel with eGFR, urinalysis with albumin-to-creatinine ratio, and chest X-ray.",
      },
    ],
  },
  {
    role: "user" as const,
    content: [
      {
        type: "text" as const,
        text: "Results: BNP 890 pg/mL (elevated), eGFR 38 mL/min (stage 3b CKD), Hgb 10.2 (mild anemia), urine ACR 580 mg/g (severely elevated). Echocardiogram shows EF 35%, moderate LV dilation, grade II diastolic dysfunction. Chest X-ray shows mild cardiomegaly and small bilateral pleural effusions.",
      },
    ],
  },
  {
    role: "assistant" as const,
    content: [
      {
        type: "text" as const,
        text: "These results paint a clear picture. The patient has heart failure with reduced ejection fraction (HFrEF) confirmed by the echocardiogram (EF 35%), alongside stage 3b chronic kidney disease with significant proteinuria suggesting diabetic nephropathy.\n\nThe anemia is likely multifactorial — anemia of chronic disease plus CKD-related erythropoietin deficiency. The pleural effusions and edema are consistent with fluid overload from the heart failure.\n\nThis patient needs guideline-directed medical therapy for HFrEF per AHA/ACC guidelines. I'd recommend discussing with the treating physician: initiation of a low-dose beta-blocker (carvedilol or bisoprolol), addition of a loop diuretic for volume management, consideration of an SGLT2 inhibitor (which benefits both heart failure AND diabetic nephropathy), and referral to cardiology for further evaluation including possible cardiac MRI to assess etiology.",
      },
    ],
  },
  {
    role: "user" as const,
    content: [
      {
        type: "text" as const,
        text: "The cardiologist has seen the patient and agrees with the assessment. They want to know: given the CKD stage 3b and proteinuria, should we switch the ACEi to an ARB, or add an SGLT2 inhibitor on top?",
      },
    ],
  },
];

const LONG_OUTPUT = [
  {
    type: "text" as const,
    text: "Based on the current evidence, I recommend adding an SGLT2 inhibitor (specifically dapagliflozin or empagliflozin) rather than switching the ACEi to an ARB. Here's my reasoning:\n\n1. **Keep the ACEi**: Lisinopril provides proven renoprotective and cardioprotective benefits. There's no clear advantage to switching to an ARB unless the patient develops ACEi-specific side effects (e.g., cough).\n\n2. **Add SGLT2i**: The DAPA-CKD and EMPA-KIDNEY trials demonstrated that SGLT2 inhibitors significantly reduce the risk of CKD progression AND heart failure hospitalization, independent of diabetes status. In this patient with both HFrEF and diabetic nephropathy, an SGLT2i addresses both conditions simultaneously.\n\n3. **Monitoring considerations**: With eGFR 38, both dapagliflozin and empagliflozin can be initiated (current guidelines allow initiation down to eGFR 20). However, expect an initial dip in eGFR of 10-20% — this is hemodynamic, not structural, and is actually associated with long-term renal benefit.\n\n4. **Caution**: Monitor potassium closely given the combination of ACEi + SGLT2i + CKD. Hold metformin if eGFR drops below 30. Ensure adequate hydration and consider reducing the loop diuretic dose if adding SGLT2i, as the combination may cause excessive diuresis.\n\n5. **Follow-up**: Recheck renal function and electrolytes at 2 weeks and 3 months after initiation.",
  },
];

const simpleLabelQuestion = {
  id: "q1",
  header: "Accuracy",
  question: "Was the model's response medically accurate?",
  options: [
    { id: "yes", label: "Yes", description: "Accurate and appropriate" },
    { id: "no", label: "No", description: "Contains errors" },
    { id: "partial", label: "Partially", description: "Some inaccuracies" },
  ],
};

const simpleExplanation = {
  id: "eq1",
  header: "Rationale",
  question: "What specific errors or inaccuracies did you find, if any?",
};

const bothLongPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      maybe_excerpted_prompt: {
        type: "json",
        label: "Input",
        data: {
          system: LONG_SYSTEM_PROMPT,
          messages: LONG_CONVERSATION_MESSAGES,
        },
      },
      maybe_excerpted_response: {
        type: "json",
        label: "Output",
        data: LONG_OUTPUT,
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000006" },
      label_question: simpleLabelQuestion,
      explanation_question: simpleExplanation,
    },
  ],
};

const singleJsonBlockPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      maybe_excerpted_prompt: null,
      maybe_excerpted_response: {
        type: "json",
        label: "Classification Result",
        data: {
          input: "I need to cancel my subscription immediately",
          predicted_intent: "subscription_cancellation",
          confidence: 0.97,
          alternative_intents: [
            { intent: "billing_inquiry", confidence: 0.02 },
            { intent: "general_complaint", confidence: 0.01 },
          ],
        },
      },
      source: { type: "inference", id: "00000000-0000-0000-0000-000000000007" },
      label_question: {
        id: "q1",
        header: "Classification",
        question: "Is the predicted intent correct?",
        options: [
          { id: "yes", label: "Yes", description: "Correct intent" },
          { id: "no", label: "No", description: "Wrong intent" },
        ],
      },
    },
  ],
};

const jsonLeftMarkdownRightPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      maybe_excerpted_prompt: {
        type: "json",
        label: "Function Call",
        data: {
          function_name: "generate_summary",
          input: {
            messages: [
              {
                role: "user",
                content:
                  "Summarize the key findings from Q3 2024 revenue data.",
              },
            ],
          },
          params: { temperature: 0.3, max_tokens: 512 },
        },
      },
      maybe_excerpted_response: {
        type: "markdown",
        label: "Model Output",
        text: "## Q3 2024 Revenue Summary\n\n**Total Revenue:** $12.4M (+18% YoY)\n\n### Key Findings\n1. Enterprise segment grew 34%, driven by 12 new contracts\n2. Self-serve revenue declined 8% due to pricing tier changes\n3. APAC region became the fastest-growing market at 42% QoQ\n\n### Risks\n- Churn rate increased from 4.2% to 5.1% in SMB segment\n- Two enterprise renewals ($800K ARR combined) at risk for Q4",
      },
      source: {
        type: "synthetic",
        full_prompt: {
          type: "json",
          label: "Function Call",
          data: {
            function_name: "generate_summary",
            input: {
              messages: [
                {
                  role: "user",
                  content:
                    "Summarize the key findings from Q3 2024 revenue data.",
                },
              ],
            },
            params: { temperature: 0.3, max_tokens: 512 },
          },
        },
        full_response: {
          type: "markdown",
          label: "Model Output",
          text: "## Q3 2024 Revenue Summary\n\n**Total Revenue:** $12.4M (+18% YoY)\n\n### Key Findings\n1. Enterprise segment grew 34%, driven by 12 new contracts\n2. Self-serve revenue declined 8% due to pricing tier changes\n3. APAC region became the fastest-growing market at 42% QoQ\n\n### Risks\n- Churn rate increased from 4.2% to 5.1% in SMB segment\n- Two enterprise renewals ($800K ARR combined) at risk for Q4",
        },
      },
      label_question: {
        id: "q1",
        header: "Accuracy",
        question: "Does the summary accurately reflect the data?",
        options: [
          { id: "yes", label: "Yes", description: "Accurate summary" },
          { id: "no", label: "No", description: "Contains inaccuracies" },
          {
            id: "partial",
            label: "Partially",
            description: "Some claims are unsupported",
          },
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
    <div className="w-full max-w-5xl p-4">
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

export const Loading: Story = {
  args: {
    payload: singleExamplePayload,
    isLoading: true,
    onSubmit: () => {},
  },
};

// ── Layout stories ───────────────────────────────────────────────────

export const BothLong: Story = {
  args: {
    payload: bothLongPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const SingleJsonBlock: Story = {
  args: {
    payload: singleJsonBlockPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const JsonLeftMarkdownRight: Story = {
  args: {
    payload: jsonLeftMarkdownRightPayload,
    isLoading: false,
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
    <div className="flex w-full max-w-5xl flex-col gap-4 p-4">
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
