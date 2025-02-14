import { Link } from "react-router";
import { Card } from "~/components/ui/card";
import {
  BookOpenText,
  SquareFunction,
  Slack,
  MessageSquare,
  Newspaper,
  Globe,
  Github,
  GalleryVerticalEnd,
  ChartSpline,
  View,
} from "lucide-react";
import {
  countInferencesByFunction,
  countEpisodes,
} from "~/utils/clickhouse/inference";
import { getConfig } from "~/utils/config/index.server";
import { useLoaderData } from "react-router";

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
      <Card className="group h-full rounded-xl border-[1px] border-gray-200 transition-colors hover:border-gray-500">
        <div className="p-6">
          <Icon className="mb-8 h-5 w-5 text-gray-500 transition-colors group-hover:text-gray-900" />
          <h3 className="text-lg font-medium">{title}</h3>
          <p className="text-xs text-gray-500">{description}</p>
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
    <Link to={source} className="group flex w-fit items-center">
      <Icon className="mr-2 h-4 w-4 text-gray-500 transition-colors group-hover:text-gray-900" />
      <span className="text-gray-700 transition-colors group-hover:text-gray-900">
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

  return {
    totalInferences,
    numFunctions,
    numEpisodes,
  };
}

export default function Home() {
  const { totalInferences, numFunctions, numEpisodes } =
    useLoaderData<typeof loader>();

  return (
    <div className="flex flex-col">
      <div className="container mx-auto my-16 max-w-[960px]">
        <div id="observability" className="mb-16">
          <h2 className="mb-1 text-2xl font-medium">Observability</h2>
          <p className="mb-6 max-w-[640px] text-sm text-gray-500">
            Monitor metrics across models and prompts and debug individual API
            calls.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/observability/inferences"
              icon={ChartSpline}
              title="Inferences"
              description={`${totalInferences.toLocaleString()} total inferences`}
            />
            <FeatureCard
              source="/observability/episodes"
              icon={GalleryVerticalEnd}
              title="Episodes"
              description={`${numEpisodes.toLocaleString()} episodes`}
            />
            <FeatureCard
              source="/observability/functions"
              icon={SquareFunction}
              title="Functions"
              description={`${numFunctions} functions`}
            />
          </div>
        </div>

        <div id="optimization" className="mb-12">
          <h2 className="mb-1 text-2xl font-medium">Optimization</h2>
          <p className="mb-6 max-w-[640px] text-sm text-gray-500">
            Optimize your prompts, models, and inference strategies.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/optimization/supervised-fine-tuning"
              icon={View}
              title="Supervised Fine-tuning"
              description={`${numFunctions} functions available`}
            />
          </div>
        </div>

        <div className="mt-16 border-t border-gray-200 pt-16">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <h3 className="mb-4 text-sm text-gray-400">Learn more</h3>
              <div className="flex flex-col gap-3">
                <FooterLink
                  source="https://www.tensorzero.com/docs"
                  icon={BookOpenText}
                >
                  Documentation
                </FooterLink>
                <FooterLink
                  source="https://github.com/tensorzero/tensorzero"
                  icon={Github}
                >
                  GitHub
                </FooterLink>
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-sm text-gray-400">Ask a question</h3>
              <div className="flex flex-col gap-3">
                <FooterLink
                  source="https://www.tensorzero.com/slack"
                  icon={Slack}
                >
                  Slack
                </FooterLink>
                <FooterLink
                  source="https://www.tensorzero.com/discord"
                  icon={MessageSquare}
                >
                  Discord
                </FooterLink>
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-sm text-gray-400">Explore TensorZero</h3>
              <div className="flex flex-col gap-3">
                <FooterLink source="https://www.tensorzero.com/" icon={Globe}>
                  Website
                </FooterLink>
                <FooterLink
                  source="https://www.tensorzero.com/blog"
                  icon={Newspaper}
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
