import { useState } from "react";
import type { InferenceByIdRow, TableBounds } from "~/utils/clickhouse";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useNavigate } from "react-router";

export default function InferencesTable({
  inferences,
  pageSize,
  bounds,
}: {
  inferences: InferenceByIdRow[];
  pageSize: number;
  bounds: TableBounds;
}) {
  const [goToId, setGoToId] = useState("");
  const navigate = useNavigate();

  // TODO: wire this to go the the details page for a particular inference, maybe add a popover.
  const handleGoTo = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const inferenceIndex = inferences.findIndex((inf) => inf.id === goToId);
    if (inferenceIndex !== -1) {
      setTimeout(() => {
        const element = document.getElementById(goToId);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "center" });
          element.classList.add("bg-yellow-100");
          setTimeout(() => element.classList.remove("bg-yellow-100"), 2000);
        }
      }, 100);
    }
    setGoToId("");
  };

  const formatDate = (date: Date) => {
    return date.toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];

  const handleNextPage = () => {
    navigate(`?before=${bottomInference.id}&page_size=${pageSize}`);
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topInference.id}&page_size=${pageSize}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious = bounds.last_id === topInference.id;
  const disableNext = bounds.first_id === bottomInference.id;

  return (
    <div>
      <h2 className="mb-4 text-2xl font-semibold">Inferences</h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <form onSubmit={handleGoTo} className="mb-4">
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="00000000-0000-0000-0000-000000000000"
            value={goToId}
            onChange={(e) => setGoToId(e.target.value)}
            className="flex-grow"
          />
          <Button type="submit">Go to Inference</Button>
        </div>
      </form>
      <div className="my-6 h-px w-full bg-gray-200"></div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Inference ID</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Function</TableHead>
            <TableHead>Variant</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {inferences.map((inference) => (
            <TableRow key={inference.id} id={inference.id}>
              <TableCell className="max-w-[200px]">
                <a href={`#${inference.id}`} className="block no-underline">
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.id}
                  </code>
                </a>
              </TableCell>
              <TableCell className="max-w-[200px]">
                <a
                  href={`#${inference.episode_id}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.episode_id}
                  </code>
                </a>
              </TableCell>
              <TableCell>
                <a
                  href={`#${inference.function_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.function_name}
                  </code>
                </a>
              </TableCell>
              <TableCell>
                <a
                  href={`#${inference.variant_name}`}
                  className="block no-underline"
                >
                  <code className="block overflow-hidden text-ellipsis whitespace-nowrap rounded font-mono transition-colors duration-300 hover:text-gray-500">
                    {inference.variant_name}
                  </code>
                </a>
              </TableCell>
              <TableCell>{formatDate(new Date(inference.timestamp))}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="mt-4 flex items-center justify-center gap-2">
        <Button
          onClick={handlePreviousPage}
          disabled={disablePrevious}
          className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          onClick={handleNextPage}
          disabled={disableNext}
          className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
