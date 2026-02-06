import { ArrowRight, BarChart3, Lightbulb, Zap } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Chat } from "~/components/icons/Icons";

const FEATURES = [
  {
    icon: BarChart3,
    title: "Analyze inferences",
    description:
      "Deep-dive into individual inferences, understand model behavior, and identify issues.",
  },
  {
    icon: Lightbulb,
    title: "Optimize functions",
    description:
      "Get actionable suggestions to improve function performance and fine-tune variants.",
  },
  {
    icon: Zap,
    title: "Automate workflows",
    description:
      "Run evaluations, manage datasets, and launch optimization jobs — all through conversation.",
  },
];

export function AutopilotUnavailableState() {
  return (
    <div className="grid min-h-full place-items-center p-8">
      <div className="flex max-w-lg flex-col items-center text-center">
        <div className="bg-muted mb-6 flex h-14 w-14 items-center justify-center rounded-full">
          <Chat className="text-fg-tertiary h-7 w-7" />
        </div>

        <h1 className="text-fg-primary text-2xl font-semibold tracking-tight">
          TensorZero Autopilot
        </h1>
        <p className="text-fg-tertiary mt-2 max-w-md text-sm text-balance">
          An AI-powered assistant that helps you analyze inferences, optimize
          functions, and automate workflows — directly from the TensorZero
          dashboard.
        </p>

        <div className="mt-8 grid w-full gap-4">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="flex items-start gap-3 rounded-lg border p-4 text-left"
            >
              <feature.icon className="text-fg-tertiary mt-0.5 h-5 w-5 shrink-0" />
              <div>
                <p className="text-fg-primary text-sm font-medium">
                  {feature.title}
                </p>
                <p className="text-fg-tertiary mt-0.5 text-sm">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 flex flex-col items-center gap-3">
          <Button asChild>
            <a
              href="https://www.tensorzero.com/autopilot"
              target="_blank"
              rel="noopener noreferrer"
            >
              Get access
              <ArrowRight className="ml-1 h-4 w-4" />
            </a>
          </Button>
          <p className="text-fg-tertiary text-xs">
            Already have access? Set{" "}
            <code className="bg-muted rounded px-1 py-0.5 font-mono">
              TENSORZERO_AUTOPILOT_API_KEY
            </code>{" "}
            on the gateway.
          </p>
        </div>
      </div>
    </div>
  );
}
