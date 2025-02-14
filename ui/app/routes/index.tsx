import { Link } from "react-router";
import { Card, CardHeader, CardTitle } from "~/components/ui/card";
import {
  BookOpenText,
  SquareFunction,
  GraduationCap,
  Slack,
  MessageSquare,
  Newspaper,
  Globe,
  Twitter,
  Github,
  GalleryVerticalEnd,
  ChartSpline,
  View,
} from "lucide-react";
import { countInferencesByFunction, countEpisodes } from "~/utils/clickhouse/inference";
import { getConfig } from "~/utils/config/index.server";
import { useLoaderData } from "react-router";

interface FeatureCardProps {
  source: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}

function FeatureCard({ source, icon: Icon, title, description }: FeatureCardProps) {
  return (
    <Link to={source} className="block">
      <Card className="h-full border-[1px] border-gray-200 rounded-xl hover:border-gray-500 transition-colors group">
        <div className="p-6">
          <Icon className="h-5 w-5 mb-8 text-gray-500 group-hover:text-gray-900 transition-colors" />
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
    <Link to={source} className="flex items-center w-fit group">
      <Icon className="mr-2 h-4 w-4 text-gray-500 group-hover:text-gray-900 transition-colors" />
      <span className="text-gray-700 group-hover:text-gray-900 transition-colors">
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
    numEpisodes
  };
}

export default function Home() {
  const { totalInferences, numFunctions, numEpisodes } = useLoaderData<typeof loader>();

  return (
    <div className="flex flex-col">
      <div className="container my-16 mx-auto max-w-[960px]">
        <div id="observability" className="mb-16">
          <h2 className="mb-1 text-2xl font-medium">Observability</h2>
          <p className="mb-6 text-sm text-gray-500 max-w-[640px]">Monitor metrics across models and prompts and debug individual API calls.</p>
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
          <p className="mb-6 text-sm text-gray-500 max-w-[640px]">Optimize your prompts, models, and inference strategies.</p>
          <div className="grid gap-6 md:grid-cols-3">
            <FeatureCard
              source="/optimization/supervised-fine-tuning"
              icon={View}
              title="Supervised Fine-tuning"
              description={`${numFunctions} functions available`}
            />
          </div>
        </div>

        <div className="mt-16 pt-16 border-t border-gray-200">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <h3 className="mb-4 text-sm text-gray-400">Learn more</h3>
              <div className="flex flex-col gap-3">
                <FooterLink source="/docs/quickstart" icon={BookOpenText}>Documentation</FooterLink>
                <FooterLink source="/docs/tutorials" icon={Github}>GitHub</FooterLink>
                <FooterLink source="/docs/concepts" icon={GraduationCap}>Tutorials</FooterLink>
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-sm text-gray-400">Ask a question</h3>
              <div className="flex flex-col gap-3">
                <FooterLink source="/docs/api/rest" icon={Slack}>Slack</FooterLink>
                <FooterLink source="/docs/api/sdk" icon={MessageSquare}>Discord</FooterLink>
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-sm text-gray-400">Explore TensorZero</h3>
              <div className="flex flex-col gap-3">
                <FooterLink source="/docs/guides" icon={Newspaper}>Blog</FooterLink>
                <FooterLink source="/docs/examples" icon={Globe}>Website</FooterLink>
                <FooterLink source="/docs/troubleshooting" icon={Twitter}>X</FooterLink>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
