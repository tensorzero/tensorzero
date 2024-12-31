import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import { Card, CardHeader, CardTitle } from "~/components/ui/card";
import { BarChart2, GitBranch, Zap, BookOpen } from "lucide-react";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-grow py-12">
        <div className="container mx-auto max-w-5xl px-4 md:px-6">
          <div id="observability" className="mb-12">
            <h2 className="mb-6 text-3xl font-bold">Observability</h2>
            <div className="grid gap-6 md:grid-cols-2">
              <Link to="#" className="block">
                <Card className="h-full transition-shadow hover:shadow-md">
                  <CardHeader>
                    <div className="flex items-center space-x-2">
                      <BarChart2 className="h-6 w-6 text-primary" />
                      <CardTitle>Inferences</CardTitle>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
              <Link to="#" className="block">
                <Card className="h-full transition-shadow hover:shadow-md">
                  <CardHeader>
                    <div className="flex items-center space-x-2">
                      <GitBranch className="h-6 w-6 text-primary" />
                      <CardTitle>Episodes</CardTitle>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </div>

          <div id="optimization" className="mb-12">
            <h2 className="mb-6 text-3xl font-bold">Optimization</h2>
            <div className="grid gap-6 md:grid-cols-2">
              <Link to="/optimization/supervised-fine-tuning" className="block">
                <Card className="h-full transition-shadow hover:shadow-md">
                  <CardHeader>
                    <div className="flex items-center space-x-2">
                      <Zap className="h-6 w-6 text-primary" />
                      <CardTitle>Fine-tuning</CardTitle>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-muted py-10">
        <div className="container mx-auto px-4 text-center md:px-6">
          <p className="mb-6 text-lg text-muted-foreground">
            Explore our documentation to get the most out of TensorZero.
          </p>
          <Button size="default" variant="default" asChild>
            <Link to="https://www.tensorzero.com/docs" target="_blank">
              View Documentation <BookOpen className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </footer>
    </div>
  );
}
