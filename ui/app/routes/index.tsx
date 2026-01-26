import { Link, type RouteHandle, Await, useAsyncError } from "react-router";
import { LayoutErrorBoundary } from "~/components/ui/error";
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
import { KeyRound } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { getConfig, getAllFunctionConfigs } from "~/utils/config/index.server";
import type { Route } from "./+types/index";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getErrorDetails } from "~/utils/tensorzero/errors";

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
      <Card className="border-border group flex w-full flex-row items-center gap-3 rounded-xl border p-4 transition-colors hover:border-orange-200">
        <div className="bg-bg-tertiary group-hover:bg-card-highlight h-8 w-8 rounded-lg p-2 transition-colors">
          <Icon
            className="text-fg-secondary group-hover:text-card-highlight-icon transition-colors"
            size={16}
          />
        </div>
        <div className="flex w-full flex-col overflow-hidden">
          <h3 className="text-fg-primary overflow-hidden text-sm font-medium text-ellipsis whitespace-nowrap transition-colors group-hover:text-orange-600">
            {title}
          </h3>
          <p className="text-fg-secondary overflow-hidden text-xs text-ellipsis whitespace-nowrap">
            {typeof description === "string" ? (
              description
            ) : (
              <React.Suspense
                fallback={
                  <span className="bg-bg-tertiary inline-block h-3 w-16 animate-pulse rounded"></span>
                }
              >
                <Await
                  resolve={description}
                  errorElement={<DirectoryCardDescriptionError />}
                >
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

function DirectoryCardDescriptionError() {
  const error = useAsyncError();
  const { message, status } = getErrorDetails(error);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help text-red-600 underline decoration-dotted">
          Error
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {message}
        {status && ` (${status})`}
      </TooltipContent>
    </Tooltip>
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
      <Icon className="text-fg-muted mr-2 h-4 w-4 transition-colors group-hover:text-orange-600" />
      <span className="text-fg-secondary text-sm transition-colors group-hover:text-orange-600">
        {children}
      </span>
    </Link>
  );
}

export async function loader() {
  const httpClient = getTensorZeroClient();

  const countsInfoPromise = httpClient.listFunctionsWithInferenceCount();
  const episodesPromise = httpClient.queryEpisodeTableBounds();
  const datasetMetadataPromise = httpClient.listDatasets({});
  const numEvaluationRunsPromise = httpClient.countEvaluationRuns();
  const numWorkflowEvaluationRunsPromise =
    httpClient.countWorkflowEvaluationRuns();
  const numWorkflowEvaluationRunProjectsPromise =
    httpClient.countWorkflowEvaluationProjects();
  const configPromise = getConfig();
  const functionConfigsPromise = getAllFunctionConfigs();
  const numModelsUsedPromise = httpClient
    .countDistinctModelsUsed()
    .then((response) => response.model_count);

  const totalInferencesDesc = countsInfoPromise.then((countsInfo) => {
    const total = countsInfo.reduce(
      (acc, curr) => acc + curr.inference_count,
      0,
    );
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

  const numDatasetsDesc = datasetMetadataPromise.then(
    (datasets) => `${datasets.datasets.length} datasets`,
  );

  const numEvaluationRunsDesc = numEvaluationRunsPromise.then(
    (runs) => `evaluations, ${runs} runs`,
  );

  const inferenceEvaluationsDesc = Promise.all([
    configPromise,
    numEvaluationRunsPromise,
  ]).then(([config, runs]) => {
    const numEvaluations = Object.keys(config.evaluations || {}).length;
    return `${numEvaluations} evaluations, ${runs} runs`;
  });

  const dynamicEvaluationsDesc = Promise.all([
    numWorkflowEvaluationRunProjectsPromise,
    numWorkflowEvaluationRunsPromise,
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
    inferenceEvaluationsDesc,
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
    inferenceEvaluationsDesc,
    dynamicEvaluationsDesc,
    numModelsUsedDesc,
  } = loaderData;

  return (
    <PageLayout>
      <div className="mx-auto flex w-full max-w-240 flex-col gap-12">
        <h1 className="text-2xl font-medium">Overview</h1>

        {/* Main sections grid */}
        {/* Row 1: Observability, Evaluations, Optimization */}
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

          <div className="flex w-full flex-col gap-12 lg:gap-6">
            <div id="evaluations" className="flex w-full flex-col gap-2">
              <h2 className="text-md text-fg-secondary font-medium">
                Evaluations
              </h2>
              <div className="flex flex-col gap-2">
                <DirectoryCard
                  source="/evaluations"
                  icon={GridCheck}
                  title="Inference Evaluations"
                  description={inferenceEvaluationsDesc}
                />
                <DirectoryCard
                  source="/workflow_evaluations"
                  icon={SequenceChecks}
                  title="Workflow Evaluations"
                  description={dynamicEvaluationsDesc}
                />
              </div>
            </div>

            <div
              id="optimization"
              className="mt-auto flex w-full flex-col gap-2"
            >
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
          </div>

          <div id="resources" className="flex w-full flex-col gap-2">
            <h2 className="text-md text-fg-secondary font-medium">Resources</h2>
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
                source="/api-keys"
                icon={KeyRound}
                title="API Keys"
                description=""
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
                Docs
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

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
