import { Link, type RouteHandle } from "react-router";
import { Card } from "~/components/ui/card";
import { PageLayout } from "~/components/layout/PageLayout";
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
  GridCheck,
  SequenceChecks,
} from "~/components/icons/Icons";
import {
  countInferencesByFunction,
  countEpisodes,
} from "~/utils/clickhouse/inference.server";
import { getAllFunctionConfigs } from "~/utils/config/index.server";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";
import { countTotalEvaluationRuns } from "~/utils/clickhouse/evaluations.server";
import { useConfig } from "~/context/config";
import type { Route } from "./+types/index";
import {
  countDynamicEvaluationProjects,
  countDynamicEvaluationRuns,
} from "~/utils/clickhouse/dynamic_evaluations.server";

export const handle: RouteHandle = {
  hideBreadcrumbs: true,
};

interface DirectoryCardProps {
  source: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}

function DirectoryCard({
  source,
  icon: Icon,
  title,
  description,
}: DirectoryCardProps) {
  return (
    <Link to={source} className="block">
      <Card className="border-border hover:border-border-hover group flex w-full flex-row items-center gap-3 rounded-xl border p-4 hover:shadow-[0_0_0_3px_rgba(0,0,0,0.05)]">
        <div className="bg-bg-tertiary h-8 w-8 rounded-lg p-2">
          <Icon className="text-fg-secondary group-hover:text-fg-primary transition-colors" />
        </div>
        <div className="flex w-full flex-col overflow-hidden">
          <h3 className="text-fg-primary overflow-hidden text-sm font-medium text-ellipsis whitespace-nowrap">
            {title}
          </h3>
          <p className="text-fg-secondary overflow-hidden text-xs text-ellipsis whitespace-nowrap">
            {description}
          </p>
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
  const [
    countsInfo,
    numEpisodes,
    datasetCounts,
    numEvaluationRuns,
    numDynamicEvaluationRuns,
    numDynamicEvaluationRunProjects,
    functionConfigs,
  ] = await Promise.all([
    countInferencesByFunction(),
    countEpisodes(),
    getDatasetCounts({}),
    countTotalEvaluationRuns(),
    countDynamicEvaluationRuns(),
    countDynamicEvaluationProjects(),
    getAllFunctionConfigs(),
  ]);
  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);
  const numFunctions = Object.keys(functionConfigs).length;
  console.log("numFunctions", numFunctions);
  const numDatasets = datasetCounts.length;

  return {
    totalInferences,
    numFunctions,
    numEpisodes,
    numDatasets,
    numEvaluationRuns,
    numDynamicEvaluationRuns,
    numDynamicEvaluationRunProjects,
  };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const {
    totalInferences,
    numFunctions,
    numEpisodes,
    numDatasets,
    numEvaluationRuns,
    numDynamicEvaluationRuns,
    numDynamicEvaluationRunProjects,
  } = loaderData;
  const config = useConfig();
  const numEvaluations = Object.keys(config.evaluations).length;

  return (
    <PageLayout>
      <div className="mx-auto flex w-full max-w-240 flex-col gap-12">
        <h1 className="text-2xl font-medium">Dashboard</h1>
        <div className="grid w-full grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
          <div id="observability" className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">
              Observability
            </h2>
            <div className="flex flex-col gap-2">
              <DirectoryCard
                source="/observability/inferences"
                icon={Inferences}
                title="Inferences"
                description={`${totalInferences.toLocaleString()} inferences`}
              />
              <DirectoryCard
                source="/observability/episodes"
                icon={Episodes}
                title="Episodes"
                description={`${numEpisodes.toLocaleString()} episodes`}
              />
              <DirectoryCard
                source="/observability/functions"
                icon={Functions}
                title="Functions"
                description={`${numFunctions} functions`}
              />
            </div>
          </div>

          <div id="optimization" className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">
              Optimization
            </h2>
            <div className="flex flex-col gap-2">
              <DirectoryCard
                source="/optimization/supervised-fine-tuning"
                icon={SupervisedFineTuning}
                title="Supervised Fine-tuning"
                description={`${numFunctions} functions`}
              />
            </div>
          </div>

          <div id="workflows" className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Workflows</h2>
            <div className="flex flex-col gap-2">
              <DirectoryCard
                source="/datasets"
                icon={Dataset}
                title="Datasets"
                description={`${numDatasets} datasets`}
              />
              <DirectoryCard
                source="/evaluations"
                icon={GridCheck}
                title="Static Evaluations"
                description={`${numEvaluations} evaluations, ${numEvaluationRuns} runs`}
              />
              <DirectoryCard
                source="/dynamic_evaluations"
                icon={SequenceChecks}
                title="Dynamic Evaluations"
                description={`${numDynamicEvaluationRunProjects} projects, ${numDynamicEvaluationRuns} runs`}
              />
            </div>
          </div>
        </div>

        <div className="border-border my-4 w-full border-t"></div>

        <div className="grid w-full grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
          <div className="w-full">
            <h3 className="text-fg-tertiary mb-4 text-sm font-medium">
              Learn more
            </h3>
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
          <div className="w-full">
            <h3 className="text-fg-tertiary mb-4 text-sm font-medium">
              Ask a question
            </h3>
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
          <div className="w-full">
            <h3 className="text-fg-tertiary mb-4 text-sm font-medium">
              Explore TensorZero
            </h3>
            <div className="flex flex-col gap-3">
              <FooterLink source="https://www.tensorzero.com/" icon={Globe}>
                Website
              </FooterLink>
              <FooterLink source="https://www.tensorzero.com/blog" icon={Blog}>
                Blog
              </FooterLink>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  );
}
