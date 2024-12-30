import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import {
  BarChart2,
  GitBranch,
  Zap,
  BookOpen,
} from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
    <main className="flex-grow py-12">
      <div className="container px-4 md:px-6">
        <div id="observability" className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Observability</h2>
          <div className="grid md:grid-cols-2 gap-6">
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
          <h2 className="text-3xl font-bold mb-6">Optimization</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Link to="/optimization/fine-tuning" className="block">
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

    <footer className="py-10 bg-muted">
      <div className="container px-4 md:px-6 text-center">
        <p className="text-lg text-muted-foreground mb-6">Explore our documentation to get the most out of TensorZero.</p>
        <Button size="default" variant="secondary" asChild>
          <Link to="https://www.tensorzero.com/docs">
            View Documentation <BookOpen className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </footer>
  </div>

  );
}