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
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Yes! Great question — it is indeed commonly used in sports and recreation. You're getting closer! Keep going with your questions.",
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
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Yes! Great question — it is indeed commonly used in sports and recreation. You're narrowing it down nicely. Keep going with your questions!",
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
                content: [
                  { type: "text", text: "Can you hold it in one hand?" },
                ],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: "No." }],
              },
              {
                role: "user",
                content: [
                  { type: "text", text: "Is it a piece of furniture?" },
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
              text: "Not exactly, but it can be a prominent piece in a room. It's more of an instrument than furniture, though some people do treat it as a decorative centerpiece in their living rooms.",
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
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Yes! It is indeed a natural feature of the Earth. You're on the right track — think big, really big. This is one of the most famous natural landmarks in the world.",
            },
          ],
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
          text: "You are a senior code reviewer with expertise in backend security and performance optimization. Review the following pull request diff carefully and identify any bugs, security vulnerabilities, or performance issues.\n\nPay special attention to:\n- Input validation and sanitization\n- SQL injection vectors\n- Resource exhaustion risks\n- Error handling gaps\n- Race conditions in concurrent code paths\n\nFor each issue found, classify its severity as CRITICAL, HIGH, MEDIUM, or LOW and provide a specific remediation recommendation.",
        },
        {
          type: "markdown",
          label: "Model Response",
          text: "## Code Review: PR #4521\n\n### Critical Issues\n\n**1. SQL Injection (CRITICAL)**\n`function_name` is user-provided input interpolated directly into the SQL query string on line 47. This allows arbitrary SQL execution. Use parameterized queries with `$1` placeholders instead of string interpolation.\n\n```sql\n-- Before (vulnerable)\nSELECT * FROM inferences WHERE function_name = '{function_name}'\n\n-- After (safe)\nSELECT * FROM inferences WHERE function_name = $1\n```\n\n**2. Unbounded batch size (HIGH)**\nNo limit on items in `batch_inferences` request body. A malicious client could send millions of items, causing OOM. Add a configurable max batch size (recommended: 1000) and return 413 Payload Too Large when exceeded.\n\n**3. Missing authentication check (HIGH)**\nThe new `/api/v1/admin/config` endpoint lacks the `require_admin` middleware that other admin endpoints use. Add `require_admin()` to the route handler chain.\n\n### Medium Issues\n\n**4. Unhandled error in async task (MEDIUM)**\nThe spawned tokio task on line 112 uses `.unwrap()` on the database result. If the insert fails, the task panics silently. Use `.expect(\"context\")` or propagate the error via a channel.\n\n**5. N+1 query pattern (MEDIUM)**\nThe loop on lines 89-95 executes a separate SELECT for each inference in the batch. Use a single `WHERE id IN (...)` query instead to reduce database round trips from O(n) to O(1).",
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
        {
          type: "json",
          label: "Output",
          data: [
            {
              type: "text",
              text: "Let me work through this step by step.\n\nThe first train has a 1-hour head start, so by 10:00 AM it has already traveled 60 miles.\n\nRemaining distance: 790 - 60 = 730 miles\n\nCombined speed after 10:00 AM: 60 + 80 = 140 mph\n\nTime to close the gap: 730 / 140 = 5.214 hours ≈ 5 hours 13 minutes\n\nMeeting time: 10:00 AM + 5h 13m = 3:13 PM",
            },
          ],
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
                    text: "Translate this legal clause from English to German, preserving all legal terminology precisely.",
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
            system:
              "You are playing 20 questions. The user is trying to guess the secret word. You must answer only yes or no. Do not give hints, elaborations, or any additional information beyond a single-word yes or no answer. The secret word is: submarine.",
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
                  {
                    type: "text",
                    text: "Is it something you'd find in a house?",
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
                    text: "Is it bigger than a person?",
                  },
                ],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: "Yes, much bigger." }],
              },
              {
                role: "user",
                content: [{ type: "text", text: "Is it a vehicle?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: "Yes." }],
              },
              {
                role: "user",
                content: [{ type: "text", text: "Does it travel on land?" }],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: "No." }],
              },
              {
                role: "user",
                content: [
                  { type: "text", text: "Does it fly through the air?" },
                ],
              },
              {
                role: "assistant",
                content: [{ type: "text", text: "No." }],
              },
              {
                role: "user",
                content: [{ type: "text", text: "Does it travel on water?" }],
              },
              {
                role: "assistant",
                content: [
                  {
                    type: "text",
                    text: "Sometimes, but it mostly travels below the surface of the water. That's its primary domain of operation.",
                  },
                ],
              },
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    text: "Is it used by the military?",
                  },
                ],
              },
              {
                role: "assistant",
                content: [
                  {
                    type: "text",
                    text: "Yes, it's heavily associated with naval military operations, though civilian versions exist for research and tourism.",
                  },
                ],
              },
              {
                role: "user",
                content: [{ type: "text", text: "Is it a submarine?" }],
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
              text: "Yes! You got it — it's a submarine! Well played, that was excellent deductive reasoning. You narrowed it down from man-made object to vehicle to watercraft in just a few questions.",
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
            headers: {
              "Content-Type": "application/json",
              Authorization: "Bearer t0-****-****",
            },
            body: {
              function_name: "classify_intent",
              input: {
                messages: [
                  {
                    role: "user",
                    content:
                      "I placed an order yesterday for two pairs of running shoes but I changed my mind. I want to cancel the entire order and get a full refund to my original payment method. My order number is ORD-2024-58291.",
                  },
                ],
              },
              params: {
                temperature: 0.1,
                max_tokens: 256,
              },
            },
          },
        },
        {
          type: "json",
          label: "API Response",
          data: {
            status: 200,
            inference_id: "inf_01H2X4Y5Z6A7B8C9D0E1F2G3H4",
            episode_id: "ep_01H2X4Y5Z6A7B8C9D0E1F2G3H5",
            variant_name: "gpt4o_classifier_v2",
            body: {
              intent: "order_cancellation",
              confidence: 0.94,
              sub_intents: ["full_refund", "payment_reversal"],
              extracted_entities: {
                order_id: "ORD-2024-58291",
                item_count: 2,
                item_category: "footwear",
              },
            },
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

// ── Edge-case fixtures (size combinations) ───────────────────────────

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

const SHORT_OUTPUT = [{ type: "text" as const, text: "Yes." }];

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

const longInputShortOutputPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: LONG_SYSTEM_PROMPT,
            messages: LONG_CONVERSATION_MESSAGES,
          },
        },
        { type: "json", label: "Output", data: SHORT_OUTPUT },
      ],
      label_question: simpleLabelQuestion,
      explanation_question: simpleExplanation,
    },
  ],
};

const shortInputLongOutputPayload: EventPayloadAutoEvalExampleLabeling = {
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
                    text: "Should we add an SGLT2 inhibitor or switch ACEi to ARB?",
                  },
                ],
              },
            ],
          },
        },
        { type: "json", label: "Output", data: LONG_OUTPUT },
      ],
      label_question: simpleLabelQuestion,
      explanation_question: simpleExplanation,
    },
  ],
};

const bothShortPayload: EventPayloadAutoEvalExampleLabeling = {
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
                content: [{ type: "text", text: "Is 2 + 2 equal to 5?" }],
              },
            ],
          },
        },
        {
          type: "json",
          label: "Output",
          data: [{ type: "text", text: "No." }],
        },
      ],
      label_question: simpleLabelQuestion,
    },
  ],
};

const bothLongPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: LONG_SYSTEM_PROMPT,
            messages: LONG_CONVERSATION_MESSAGES,
          },
        },
        { type: "json", label: "Output", data: LONG_OUTPUT },
      ],
      label_question: simpleLabelQuestion,
      explanation_question: simpleExplanation,
    },
  ],
};

const LONG_MARKDOWN_PROMPT =
  "You are a senior technical writer. Review the following API documentation draft for completeness, accuracy, and clarity.\n\nCheck for:\n- Missing parameters or return values\n- Incorrect HTTP methods or status codes\n- Ambiguous descriptions that could confuse developers\n- Missing error handling documentation\n- Inconsistent naming conventions\n- Missing authentication/authorization details\n- Lack of request/response examples\n\nProvide your review as a structured list of issues with severity levels (CRITICAL, MAJOR, MINOR, SUGGESTION).";

const LONG_MARKDOWN_RESPONSE =
  "## Documentation Review: `/api/v1/inference` Endpoint\n\n### CRITICAL Issues\n\n**1. Missing authentication documentation**\nThe endpoint requires a Bearer token but the docs don't mention this. Developers will get 401 errors with no guidance on how to authenticate.\n\n**Recommendation:** Add an \"Authentication\" section before the parameters table:\n```\nAuthentication: Bearer token required\nHeader: Authorization: Bearer <your-api-key>\n```\n\n**2. Incorrect HTTP method for batch endpoint**\nThe batch inference endpoint is documented as `GET /api/v1/batch_inference` but it should be `POST` since it accepts a request body with input data.\n\n### MAJOR Issues\n\n**3. Missing error response documentation**\nOnly 200 and 500 are documented. Missing:\n- `400 Bad Request` — invalid input format, missing required fields\n- `401 Unauthorized` — missing or invalid API key\n- `404 Not Found` — unknown function name\n- `413 Payload Too Large` — batch size exceeds limit\n- `429 Too Many Requests` — rate limit exceeded\n\n**4. Ambiguous `input` parameter type**\nThe `input` field is described as \"object\" but it can be either:\n- A chat-format object with `messages` array\n- A JSON-format object with arbitrary structure matching the function's schema\n\nThese have completely different validation rules and should be documented separately.\n\n**5. Missing `variant_name` in response schema**\nThe response includes `variant_name` indicating which model variant served the request, but this field isn't in the documented response schema.\n\n### MINOR Issues\n\n**6. Inconsistent parameter naming**\nThe request uses `function_name` (snake_case) but the response uses `inferenceId` (camelCase). Pick one convention.\n\n**7. Missing pagination docs for list endpoints**\nThe `/api/v1/inferences` list endpoint mentions pagination but doesn't document the `cursor`, `limit`, or `order` parameters.\n\n### SUGGESTIONS\n\n**8. Add request/response examples**\nInclude at least one complete curl example with headers, request body, and expected response for each endpoint.\n\n**9. Add rate limit headers documentation**\nDocument `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` response headers.";

const longMarkdownBothPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        { type: "markdown", label: "Prompt", text: LONG_MARKDOWN_PROMPT },
        {
          type: "markdown",
          label: "Model Response",
          text: LONG_MARKDOWN_RESPONSE,
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
      explanation_question: {
        id: "eq1",
        header: "Details",
        question: "Which issues were incorrect or miscategorized?",
      },
    },
  ],
};

const shortMarkdownBothPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "markdown",
          label: "Prompt",
          text: "Translate 'hello' to French.",
        },
        { type: "markdown", label: "Model Response", text: "Bonjour." },
      ],
      label_question: {
        id: "q1",
        header: "Accuracy",
        question: "Is the translation correct?",
        options: [
          { id: "yes", label: "Yes", description: "Correct translation" },
          { id: "no", label: "No", description: "Incorrect translation" },
        ],
      },
    },
  ],
};

const longMarkdownShortResponsePayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        { type: "markdown", label: "Prompt", text: LONG_MARKDOWN_PROMPT },
        {
          type: "markdown",
          label: "Model Response",
          text: "Looks good to me. No issues found.",
        },
      ],
      label_question: {
        id: "q1",
        header: "Thoroughness",
        question: "Was the review sufficiently thorough?",
        options: [
          { id: "yes", label: "Yes", description: "Thorough review" },
          { id: "no", label: "No", description: "Insufficiently thorough" },
        ],
      },
    },
  ],
};

// ── Cross-type and single-block edge cases ────────────────────────────

const jsonLeftMarkdownRightPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
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
        {
          type: "markdown",
          label: "Model Output",
          text: "## Q3 2024 Revenue Summary\n\n**Total Revenue:** $12.4M (+18% YoY)\n\n### Key Findings\n1. Enterprise segment grew 34%, driven by 12 new contracts\n2. Self-serve revenue declined 8% due to pricing tier changes\n3. APAC region became the fastest-growing market at 42% QoQ\n\n### Risks\n- Churn rate increased from 4.2% to 5.1% in SMB segment\n- Two enterprise renewals ($800K ARR combined) at risk for Q4",
        },
      ],
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

const markdownLeftJsonRightPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "markdown",
          label: "Instructions",
          text: "Extract all named entities from the following customer support ticket and return them as structured JSON.\n\nEntity types to extract:\n- **person**: Customer or agent names\n- **product**: Product names or SKUs\n- **date**: Dates or time references\n- **order_id**: Order identifiers\n- **amount**: Monetary amounts",
        },
        {
          type: "json",
          label: "Model Output",
          data: {
            entities: [
              { type: "person", value: "Sarah Chen", span: [12, 22] },
              {
                type: "product",
                value: "UltraFit Pro Running Shoes",
                span: [45, 71],
              },
              { type: "date", value: "March 3rd", span: [89, 98] },
              { type: "order_id", value: "ORD-2024-71832", span: [112, 126] },
              { type: "amount", value: "$149.99", span: [140, 147] },
            ],
            confidence: 0.92,
          },
        },
      ],
      label_question: {
        id: "q1",
        header: "Extraction Quality",
        question: "Were all entities correctly extracted and typed?",
        options: [
          { id: "yes", label: "Yes", description: "All correct" },
          { id: "no", label: "No", description: "Missing or incorrect" },
        ],
      },
      explanation_question: {
        id: "eq1",
        header: "Details",
        question: "Which entities were missed or misclassified?",
      },
    },
  ],
};

const singleJsonBlockPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
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
      ],
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

const singleMarkdownBlockPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "markdown",
          label: "Generated Documentation",
          text: '# `POST /api/v1/inference`\n\nRun a single inference against a configured function.\n\n## Request Body\n\n| Field | Type | Required | Description |\n|-------|------|----------|-------------|\n| `function_name` | string | Yes | Name of the function to invoke |\n| `input` | object | Yes | Input payload matching the function schema |\n| `params` | object | No | Override model parameters (temperature, max_tokens) |\n| `stream` | boolean | No | Enable streaming response (default: false) |\n\n## Response\n\n```json\n{\n  "inference_id": "inf_01H2X4Y5Z6A7B8C9",\n  "episode_id": "ep_01H2X4Y5Z6A7B8C9",\n  "variant_name": "gpt4o_v2",\n  "content": [\n    { "type": "text", "text": "..." }\n  ]\n}\n```\n\n## Error Codes\n\n- `400` — Invalid input format\n- `404` — Unknown function name\n- `429` — Rate limit exceeded\n- `500` — Internal server error',
        },
      ],
      label_question: {
        id: "q1",
        header: "Completeness",
        question: "Is the documentation complete and accurate?",
        options: [
          { id: "yes", label: "Yes", description: "Complete and accurate" },
          { id: "no", label: "No", description: "Missing or incorrect info" },
          {
            id: "partial",
            label: "Partially",
            description: "Has gaps but mostly correct",
          },
        ],
      },
    },
  ],
};

const longJsonShortMarkdownPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            system: LONG_SYSTEM_PROMPT,
            messages: LONG_CONVERSATION_MESSAGES,
          },
        },
        {
          type: "markdown",
          label: "Model Response",
          text: "Agree with the cardiologist's assessment. Add SGLT2i.",
        },
      ],
      label_question: simpleLabelQuestion,
      explanation_question: simpleExplanation,
    },
  ],
};

const shortJsonLongMarkdownPayload: EventPayloadAutoEvalExampleLabeling = {
  examples: [
    {
      context: [
        {
          type: "json",
          label: "Input",
          data: {
            messages: [{ role: "user", content: "Review the API docs" }],
          },
        },
        {
          type: "markdown",
          label: "Model Response",
          text: LONG_MARKDOWN_RESPONSE,
        },
      ],
      label_question: {
        id: "q1",
        header: "Quality",
        question: "Is the review thorough and accurate?",
        options: [
          { id: "yes", label: "Yes", description: "Thorough review" },
          { id: "no", label: "No", description: "Superficial or inaccurate" },
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

// ── Edge-case stories (size combinations) ────────────────────────────

export const LongInputShortOutput: Story = {
  args: {
    payload: longInputShortOutputPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const ShortInputLongOutput: Story = {
  args: {
    payload: shortInputLongOutputPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const BothShort: Story = {
  args: {
    payload: bothShortPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const BothLong: Story = {
  args: {
    payload: bothLongPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MarkdownBothLong: Story = {
  args: {
    payload: longMarkdownBothPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MarkdownBothShort: Story = {
  args: {
    payload: shortMarkdownBothPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const MarkdownLongPromptShortResponse: Story = {
  args: {
    payload: longMarkdownShortResponsePayload,
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

export const MarkdownLeftJsonRight: Story = {
  args: {
    payload: markdownLeftJsonRightPayload,
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

export const SingleMarkdownBlock: Story = {
  args: {
    payload: singleMarkdownBlockPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const LongJsonShortMarkdown: Story = {
  args: {
    payload: longJsonShortMarkdownPayload,
    isLoading: false,
    onSubmit: () => {},
  },
};

export const ShortJsonLongMarkdown: Story = {
  args: {
    payload: shortJsonLongMarkdownPayload,
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
