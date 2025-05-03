import { Link } from "react-router";
import { Card } from "~/components/ui/card";
import {
  Inferences,
  Episodes,
  Functions,
  SupervisedFineTuning,
  Blog,
  Discord,
  Slack,
  GitHub,
  Globe,
  Documentation,
  Dataset,
  Evaluation,
} from "~/components/icons/Icons";
import {
  countInferencesByFunction,
  countEpisodes,
} from "~/utils/clickhouse/inference.server";
import { getConfig } from "~/utils/config/index.server";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";
import { countTotalEvaluationRuns } from "~/utils/clickhouse/evaluations.server";
import { useConfig } from "~/context/config";
import type { Route } from "./+types/index";

interface FeatureCardProps {
  source: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}

function FeatureCard({
  source,
  icon: Icon,
  title,
  description,
}: FeatureCardProps) {
  return (
    <Link to={source} className="block">
      <Card className="hover:border-border-hover group border-border h-full rounded-xl border-[1px] hover:shadow-[0_0_0_4px_rgba(0,0,0,0.05)]">
        <div className="p-6">
          <Icon className="text-fg-secondary group-hover:text-foreground mb-8 h-4 w-4 transition-colors" />
          <h3 className="text-lg font-medium">{title}</h3>
          <p className="text-fg-secondary text-xs">{description}</p>
        </div>
      </Card>
    </Link>
  );
}

interface FooterLinkProps {
  source: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}

function FooterLink({ source, icon: Icon, children }: FooterLinkProps) {
  return (
    <Link
      to={source}
      className="group flex w-fit items-center"
      rel="noopener noreferrer"
      target="_blank"
    >
      <Icon className="text-fg-muted group-hover:text-fg-secondary mr-2 h-4 w-4 transition-colors" />
      <span className="text-fg-secondary group-hover:text-fg-primary transition-colors">
        {children}
      </span>
    </Link>
  );
}

export async function loader() {
  const countsInfo = await countInferencesByFunction();
  const config = await getConfig();
  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);
  const numFunctions = Object.keys(config.functions).length;
  const numEpisodes = await countEpisodes();
  const datasetCounts = await getDatasetCounts();
  const numDatasets = datasetCounts.length;
  const numEvaluationRuns = await countTotalEvaluationRuns();

  return {
    totalInferences,
    numFunctions,
    numEpisodes,
    numDatasets,
    numEvaluationRuns,
  };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const {
    totalInferences,
    numFunctions,
    numEpisodes,
    numDatasets,
    numEvaluationRuns,
  } = loaderData;
  const config = useConfig();
  const numEvaluations = Object.keys(config.evaluations).length;

  return (
    <div className="flex flex-col">
      <div className="container mx-auto my-16 max-w-[960px]">
        <div id="observability" className="mb-16">
          <h2 className="mb-1 text-2xl font-medium">Observability</h2>
          <p className="text-fg-tertiary mb-6 max-w-[640px] text-sm">
            Monitor metrics across models and prompts and debug individual API
            calls.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/observability/inferences"
              icon={Inferences}
              title="Inferences"
              description={`${totalInferences.toLocaleString()} inferences`}
            />
            <FeatureCard
              source="/observability/episodes"
              icon={Episodes}
              title="Episodes"
              description={`${numEpisodes.toLocaleString()} episodes`}
            />
            <FeatureCard
              source="/observability/functions"
              icon={Functions}
              title="Functions"
              description={`${numFunctions} functions`}
            />
          </div>
        </div>

        <div id="optimization" className="mb-16">
          <h2 className="mb-1 text-2xl font-medium">Optimization</h2>
          <p className="text-fg-tertiary mb-6 max-w-[640px] text-sm">
            Optimize your prompts, models, and inference strategies.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/optimization/supervised-fine-tuning"
              icon={SupervisedFineTuning}
              title="Supervised Fine-tuning"
              description={`${numFunctions} functions`}
            />
          </div>
        </div>

        <div id="workflows" className="mb-12">
          <h2 className="mb-1 text-2xl font-medium">Workflows</h2>
          <p className="text-fg-tertiary mb-6 max-w-[640px] text-sm">
            Manage your LLM engineering workflows.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/datasets"
              icon={Dataset}
              title="Datasets"
              description={`${numDatasets} datasets`}
            />
            <FeatureCard
              source="/evaluations"
              icon={Evaluation}
              title="Evaluations"
              description={`${numEvaluations} evaluations, ${numEvaluationRuns} runs`}
            />
          </div>
        </div>

        <div className="mt-16 border-t border-gray-200 pt-16">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <h3 className="text-fg-secondary mb-4 text-sm">Learn more</h3>
              <div className="flex flex-col gap-3">
                <FooterLink
                  source="https://www.tensorzero.com/docs"
                  icon={Documentation}
                >
                  Documentation
                </FooterLink>
                <FooterLink
                  source="https://github.com/tensorzero/tensorzero"
                  icon={GitHub}
                >
                  GitHub
                </FooterLink>
              </div>
            </div>

            <div>
              <h3 className="text-fg-secondary mb-4 text-sm">Ask a question</h3>
              <div className="flex flex-col gap-3">
                <FooterLink
                  source="https://www.tensorzero.com/slack"
                  icon={Slack}
                >
                  Slack
                </FooterLink>
                <FooterLink
                  source="https://www.tensorzero.com/discord"
                  icon={Discord}
                >
                  Discord
                </FooterLink>
              </div>
            </div>

            <div>
              <h3 className="text-fg-secondary mb-4 text-sm">
                Explore TensorZero
              </h3>
              <div className="flex flex-col gap-3">
                <FooterLink source="https://www.tensorzero.com/" icon={Globe}>
                  Website
                </FooterLink>
                <FooterLink
                  source="https://www.tensorzero.com/blog"
                  icon={Blog}
                >
                  Blog
                </FooterLink>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
