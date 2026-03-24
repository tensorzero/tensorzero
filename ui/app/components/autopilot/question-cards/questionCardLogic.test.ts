import { describe, it, expect } from "vitest";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";
import {
  StepStatus,
  applyMcToggle,
  applyOtherToggle,
  applyMcFreeTextChange,
  isStepAnswered,
  buildResponses,
  getStep,
  markUnansweredAsSkipped,
} from "./questionCardLogic";
import type { StepAnswers } from "./questionCardLogic";

const singleSelectQuestion: EventPayloadUserQuestion = {
  id: "q1",
  header: "Strategy",
  question: "Pick one",
  type: "multiple_choice",
  options: [
    { id: "a", label: "Option A", description: "desc A" },
    { id: "b", label: "Option B", description: "desc B" },
  ],
  multi_select: false,
  include_free_response: true,
};

const multiSelectQuestion: EventPayloadUserQuestion = {
  id: "q2",
  header: "Features",
  question: "Pick many",
  type: "multiple_choice",
  options: [
    { id: "x", label: "Feature X", description: "desc X" },
    { id: "y", label: "Feature Y", description: "desc Y" },
  ],
  multi_select: true,
  include_free_response: true,
};

function emptyAnswers(): StepAnswers {
  return new Map();
}

describe("single-select with Other", () => {
  it("selecting a predefined option deselects Other and clears free text", () => {
    // Start with Other selected and some text
    let answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "my custom answer",
        },
      ],
    ]);

    // Select option "a"
    answers = applyMcToggle(answers, 0, "a", singleSelectQuestion);
    const step = getStep(answers, 0);
    expect(step.status).toBe(StepStatus.AnsweredMultipleChoice);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.selected).toEqual(new Set(["a"]));
    expect(step.otherSelected).toBe(false);
    expect(step.freeResponseText).toBe("");
  });

  it("selecting Other deselects predefined options", () => {
    let answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["a"]),
          otherSelected: false,
          freeResponseText: "",
        },
      ],
    ]);

    answers = applyOtherToggle(answers, 0, singleSelectQuestion);
    const step = getStep(answers, 0);
    expect(step.status).toBe(StepStatus.AnsweredMultipleChoice);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.selected.size).toBe(0);
    expect(step.otherSelected).toBe(true);
  });

  it("toggling Other off clears free text", () => {
    let answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "my answer",
        },
      ],
    ]);

    answers = applyOtherToggle(answers, 0, singleSelectQuestion);
    const step = getStep(answers, 0);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.otherSelected).toBe(false);
    expect(step.freeResponseText).toBe("");
  });
});

describe("multi-select with Other", () => {
  it("selecting a predefined option preserves Other state", () => {
    let answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["x"]),
          otherSelected: true,
          freeResponseText: "custom",
        },
      ],
    ]);

    answers = applyMcToggle(answers, 0, "y", multiSelectQuestion);
    const step = getStep(answers, 0);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.selected).toEqual(new Set(["x", "y"]));
    expect(step.otherSelected).toBe(true);
    expect(step.freeResponseText).toBe("custom");
  });

  it("toggling Other does not clear predefined selections", () => {
    let answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["x"]),
          otherSelected: false,
          freeResponseText: "",
        },
      ],
    ]);

    answers = applyOtherToggle(answers, 0, multiSelectQuestion);
    const step = getStep(answers, 0);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.selected).toEqual(new Set(["x"]));
    expect(step.otherSelected).toBe(true);
  });
});

describe("isStepAnswered", () => {
  it("returns false for unanswered step", () => {
    expect(isStepAnswered(emptyAnswers(), 0)).toBe(false);
  });

  it("returns true when a predefined option is selected", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["a"]),
          otherSelected: false,
          freeResponseText: "",
        },
      ],
    ]);
    expect(isStepAnswered(answers, 0)).toBe(true);
  });

  it("returns true when Other is selected with text", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "my answer",
        },
      ],
    ]);
    expect(isStepAnswered(answers, 0)).toBe(true);
  });

  it("returns false when Other is selected but text is empty", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "  ",
        },
      ],
    ]);
    expect(isStepAnswered(answers, 0)).toBe(false);
  });
});

describe("applyMcFreeTextChange", () => {
  it("updates free response text while preserving other state", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["x"]),
          otherSelected: true,
          freeResponseText: "",
        },
      ],
    ]);

    const updated = applyMcFreeTextChange(answers, 0, "new text");
    const step = getStep(updated, 0);
    if (step.status !== StepStatus.AnsweredMultipleChoice) return;
    expect(step.selected).toEqual(new Set(["x"]));
    expect(step.otherSelected).toBe(true);
    expect(step.freeResponseText).toBe("new text");
  });
});

describe("markUnansweredAsSkipped", () => {
  it("preserves free-text-only multiple choice answers", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "my custom answer",
        },
      ],
    ]);

    const finalAnswers = markUnansweredAsSkipped(
      [singleSelectQuestion],
      answers,
    );
    expect(getStep(finalAnswers, 0)).toEqual(getStep(answers, 0));
  });

  it("marks empty multiple choice other answers as skipped", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "   ",
        },
      ],
    ]);

    const finalAnswers = markUnansweredAsSkipped(
      [singleSelectQuestion],
      answers,
    );
    expect(getStep(finalAnswers, 0)).toEqual({ status: StepStatus.Skipped });
  });
});

describe("buildResponses", () => {
  it("includes free_response_text when Other text is present", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set<string>(),
          otherSelected: true,
          freeResponseText: "my custom answer",
        },
      ],
    ]);

    const responses = buildResponses([singleSelectQuestion], answers);
    expect(responses["q1"]).toEqual({
      type: "multiple_choice",
      selected: [],
      free_response_text: "my custom answer",
    });
  });

  it("omits free_response_text when empty", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["a"]),
          otherSelected: false,
          freeResponseText: "",
        },
      ],
    ]);

    const responses = buildResponses([singleSelectQuestion], answers);
    expect(responses["q1"]).toEqual({
      type: "multiple_choice",
      selected: ["a"],
    });
  });

  it("includes both selected options and free text for multi-select", () => {
    const answers: StepAnswers = new Map([
      [
        0,
        {
          status: StepStatus.AnsweredMultipleChoice,
          selected: new Set(["x", "y"]),
          otherSelected: true,
          freeResponseText: "also this",
        },
      ],
    ]);

    const responses = buildResponses([multiSelectQuestion], answers);
    const response = responses["q2"];
    expect(response.type).toBe("multiple_choice");
    if (response.type !== "multiple_choice") return;
    expect(response.selected).toContain("x");
    expect(response.selected).toContain("y");
    expect(response.free_response_text).toBe("also this");
  });
});
