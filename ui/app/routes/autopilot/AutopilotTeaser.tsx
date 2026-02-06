import { ArrowRight } from "lucide-react";
import { Chat } from "~/components/icons/Icons";

const CAPABILITIES = [
  "Analyze observability data to surface error patterns and optimization opportunities",
  "Generate and refine prompts based on human feedback, metrics, and evaluations",
  "Drive optimization workflows like fine-tuning, reinforcement learning, and distillation",
  "Set up evaluations, run A/B tests, and close the feedback loop",
  "Update your TensorZero configuration with new variants, evaluations, and experiments",
  "And much more — if you can describe it, Autopilot can do it",
];


export function AutopilotTeaser() {
  return (
    <div className="flex min-h-[calc(100vh-2rem)] items-center justify-center px-8 pb-8">
      <div className="mx-auto flex max-w-2xl flex-col items-center gap-10 text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-orange-50">
          <Chat className="h-10 w-10 text-orange-600" />
        </div>

        <div className="flex flex-col gap-4">
          <h2 className="text-fg-primary text-2xl font-semibold">
            Automated AI engineer for your LLM systems
          </h2>
          <p className="text-fg-secondary max-w-[34rem] text-base leading-relaxed">
            TensorZero Autopilot analyzes your inference data, optimizes prompts
            and models, sets up evals, runs A/B tests — and handles the rest.
          </p>
        </div>

        <div className="flex max-w-md flex-col gap-3 text-left">
          {CAPABILITIES.map((capability) => (
            <div
              key={capability}
              className="text-fg-secondary flex items-start gap-3 text-sm"
            >
              <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-orange-600" />
              <span>{capability}</span>
            </div>
          ))}
        </div>

        <a
          href="https://www.tensorzero.com/autopilot-waitlist"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center rounded-lg bg-orange-600 px-8 py-3 text-base font-medium text-white transition-colors hover:bg-orange-700"
        >
          Join the waitlist
        </a>
      </div>
    </div>
  );
}
