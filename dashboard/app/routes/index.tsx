import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "~/components/ui/card";
import {
  BarChart2,
  GitBranch,
  Zap,
  BookOpen,
  ArrowRight,
  Clock,
  Sparkles,
} from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-background to-background/80">
      <main className="flex-grow py-8  px-4 flex items-center">
        <div className="max-w-6xl mx-auto w-full">
          <div className="text-center  mb-16">
            <div className="flex items-center justify-center mb-4">
              <div className="flex items-center gap-2">
                <div className="flex aspect-square w-8 h-8 items-center justify-center rounded-lg bg-primary/5">
                  <img
                    src="https://www.tensorzero.com/favicon.svg"
                    alt="TensorZero logo"
                    className="w-6 h-6"
                  />
                </div>
                <h1 className="text-3xl font-bold">TensorZero</h1>
              </div>
            </div>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Create a feedback loop for optimizing LLM applications — turn
              production data into smarter, faster, and cheaper models
            </p>
          </div>

          <div id="observability" className="my-12">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <span className="bg-primary/10 p-1.5 rounded-lg">
                <BarChart2 className="h-5 w-5 text-primary" />
              </span>
              Observability
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <Link to="#" className="block group">
                <Card className="h-full transition-all hover:shadow-md hover:border-primary/20">
                  <CardHeader className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-1.5 rounded-lg bg-primary/10">
                        <Clock className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-lg mb-1">Inferences</CardTitle>
                        <CardDescription className="text-sm">
                          Monitor inference & feedback metrics with &lt;1ms P99 overhead
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
              <Link to="#" className="block group">
                <Card className="h-full transition-all hover:shadow-md hover:border-primary/20">
                  <CardHeader className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-1.5 rounded-lg bg-primary/10">
                        <GitBranch className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-lg mb-1">Episodes</CardTitle>
                        <CardDescription className="text-sm">
                          Track multi-step LLM systems and interactions
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </div>

          <div id="optimization" className="mb-8">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <span className="bg-primary/10 p-1.5 rounded-lg">
                <Sparkles className="h-5 w-5 text-primary" />
              </span>
              Optimization
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <Link to="/optimization/fine-tuning" className="block group">
                <Card className="h-full transition-all hover:shadow-md hover:border-primary/20">
                  <CardHeader className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-1.5 rounded-lg bg-primary/10">
                        <Zap className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-lg mb-1">Fine-tuning</CardTitle>
                        <CardDescription className="text-sm">
                          Optimize prompts, models, and inference strategies using production data
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </div>
        </div>
      </main>

      <footer className="py-6 bg-muted/50 flex items-center justify-center border-t">
        <div className="container max-w-6xl text-center">
          <p className="text-sm text-muted-foreground mb-4">
            Get started with our comprehensive documentation
          </p>
          <div className="flex justify-center gap-4">
            <a target="_blank" href="https://www.tensorzero.com/docs/gateway/quickstart">
              <Button variant="outline" size="sm" className="gap-2">
                Quick Start (5min) <ArrowRight className="h-4 w-4" />
              </Button>
            </a>
            <a target="_blank" href="https://www.tensorzero.com/docs">
              <Button variant="secondary" size="sm" className="gap-2">
                Documentation <BookOpen className="h-4 w-4" />
              </Button>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}