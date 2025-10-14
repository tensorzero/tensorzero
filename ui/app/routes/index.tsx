import { Link, type RouteHandle, Await } from "react-router";
import * as React from "react";
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
  Playground,
  Model,
} from "~/components/icons/Icons";
import { countInferencesByFunction } from "~/utils/clickhouse/inference.server";
import { getConfig, getAllFunctionConfigs } from "~/utils/config/index.server";
import { getDatasetMetadata } from "~/utils/clickhouse/datasets.server";
import { countTotalEvaluationRuns } from "~/utils/clickhouse/evaluations.server";
import type { Route } from "./+types/index";
import {
  countDynamicEvaluationProjects,
  countDynamicEvaluationRuns,
} from "~/utils/clickhouse/dynamic_evaluations.server";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";

export const handle: RouteHandle = {
  hideBreadcrumbs: true,
};

interface DirectoryCardProps {
  source: string;
  icon: React.ComponentType<{ className?: string; size?: number }>;
  title: string;
  description: string | Promise<string>;
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
          <Icon
            className="text-fg-secondary group-hover:text-fg-primary transition-colors"
            size={16}
          />
        </div>
        <div className="flex w-full flex-col overflow-hidden">
          <h3 className="text-fg-primary overflow-hidden text-ellipsis whitespace-nowrap text-sm font-medium">
            {title}
          </h3>
          <p className="text-fg-secondary overflow-hidden text-ellipsis whitespace-nowrap text-xs">
            {typeof description === "string" ? (
              description
            ) : (
              <React.Suspense
                fallback={
                  <span className="bg-bg-tertiary inline-block h-3 w-16 animate-pulse rounded"></span>
                }
              >
                <Await resolve={description}>
                  {(resolvedDescription) => resolvedDescription}
                </Await>
              </React.Suspense>
            )}
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
  const nativeDatabaseClient = await getNativeDatabaseClient();

  // Create the promises
  const countsInfoPromise = countInferencesByFunction();
  const episodesPromise = nativeDatabaseClient.queryEpisodeTableBounds();
  const datasetMetadata = getDatasetMetadata({});
  const numEvaluationRunsPromise = countTotalEvaluationRuns();
  const numDynamicEvaluationRunsPromise = countDynamicEvaluationRuns();
  const numDynamicEvaluationRunProjectsPromise =
    countDynamicEvaluationProjects();
  const configPromise = getConfig();
  const functionConfigsPromise = getAllFunctionConfigs();
  const numModelsUsedPromise = nativeDatabaseClient.countDistinctModelsUsed();

  // Create derived promises - these will be stable references
  const totalInferencesDesc = countsInfoPromise.then((countsInfo) => {
    const total = countsInfo.reduce((acc, curr) => acc + curr.count, 0);
    return `${total.toLocaleString()} inferences`;
  });

  const numFunctionsDesc = functionConfigsPromise.then((functionConfigs) => {
    const numFunctions = Object.keys(functionConfigs).length;
    return `${numFunctions} functions`;
  });

  const numVariantsDesc = functionConfigsPromise.then((functionConfigs) => {
    const numVariants = Object.values(functionConfigs).reduce(
      (acc, funcConfig) => {
        return (
          acc + (funcConfig ? Object.keys(funcConfig.variants || {}).length : 0)
        );
      },
      0,
    );
    return `${numVariants} variants`;
  });

  const numEpisodesDesc = episodesPromise.then(
    (result) => `${result.count.toLocaleString()} episodes`,
  );

  const numDatasetsDesc = datasetMetadata.then(
    (datasetCounts) => `${datasetCounts.length} datasets`,
  );

  const numEvaluationRunsDesc = numEvaluationRunsPromise.then(
    (runs) => `evaluations, ${runs} runs`,
  );

  // We need to create a special promise for the static evaluations that includes the config count
  const staticEvaluationsDesc = Promise.all([
    configPromise,
    numEvaluationRunsPromise,
  ]).then(([config, runs]) => {
    const numEvaluations = Object.keys(config.evaluations || {}).length;
    return `${numEvaluations} evaluations, ${runs} runs`;
  });

  const dynamicEvaluationsDesc = Promise.all([
    numDynamicEvaluationRunProjectsPromise,
    numDynamicEvaluationRunsPromise,
  ]).then(([projects, runs]) => `${projects} projects, ${runs} runs`);

  const numModelsUsedDesc = numModelsUsedPromise.then(
    (numModelsUsed) => `${numModelsUsed} models used`,
  );

  return {
    totalInferencesDesc,
    numFunctionsDesc,
    numVariantsDesc,
    numEpisodesDesc,
    numDatasetsDesc,
    numEvaluationRunsDesc,
    staticEvaluationsDesc,
    dynamicEvaluationsDesc,
    numModelsUsedDesc,
  };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const {
    totalInferencesDesc,
    numFunctionsDesc,
    numVariantsDesc,
    numEpisodesDesc,
    numDatasetsDesc,
    staticEvaluationsDesc,
    dynamicEvaluationsDesc,
    numModelsUsedDesc,
  } = loaderData;

  return (
    <PageLayout>
      <div className="max-w-240 mx-auto flex w-full flex-col gap-12">
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
                description={totalInferencesDesc}
              />
              <DirectoryCard
                source="/observability/episodes"
                icon={Episodes}
                title="Episodes"
                description={numEpisodesDesc}
              />
              <DirectoryCard
                source="/observability/functions"
                icon={Functions}
                title="Functions"
                description={numFunctionsDesc}
              />
              <DirectoryCard
                source="/observability/models"
                icon={Model}
                title="Models"
                description={numModelsUsedDesc}
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
                description={numFunctionsDesc}
              />
            </div>
          </div>

          <div id="workflows" className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Workflows</h2>
            <div className="flex flex-col gap-2">
              <DirectoryCard
                source="/playground"
                icon={Playground}
                title="Playground"
                description={numVariantsDesc}
              />
              <DirectoryCard
                source="/datasets"
                icon={Dataset}
                title="Datasets"
                description={numDatasetsDesc}
              />
              <DirectoryCard
                source="/evaluations"
                icon={GridCheck}
                title="Static Evaluations"
                description={staticEvaluationsDesc}
              />
              <DirectoryCard
                source="/dynamic_evaluations"
                icon={SequenceChecks}
                title="Dynamic Evaluations"
                description={dynamicEvaluationsDesc}
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
