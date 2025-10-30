import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import type { KeyInfo } from "tensorzero-node";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { formatDate } from "~/utils/date";

function ApiKeyRow({ apiKey }: { apiKey: KeyInfo }) {
  const isDisabled = apiKey.disabled_at !== null;

  const publicIdElement = (
    <code
      className={`inline-block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap ${
        isDisabled ? "cursor-help line-through" : ""
      }`}
    >
      {apiKey.public_id}
    </code>
  );

  return (
    <TableRow
      key={apiKey.public_id}
      id={apiKey.public_id}
      className={isDisabled ? "opacity-50" : ""}
    >
      <TableCell className="w-0">
        {isDisabled ? (
          <Tooltip>
            <TooltipTrigger asChild>{publicIdElement}</TooltipTrigger>
            <TooltipContent>
              Disabled on {formatDate(new Date(apiKey.disabled_at!))}
            </TooltipContent>
          </Tooltip>
        ) : (
          publicIdElement
        )}
      </TableCell>
      <TableCell>
        {apiKey.description ? (
          <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
            {apiKey.description}
          </span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
      </TableCell>
      <TableCell className="w-0">
        <span className="whitespace-nowrap">
          {apiKey.created_at}
          {/*{formatDate(new Date(apiKey.created_at))}*/}
        </span>
      </TableCell>
    </TableRow>
  );
}

export default function AuthTable({ apiKeys }: { apiKeys: KeyInfo[] }) {
  return (
    <TooltipProvider delayDuration={400}>
      <div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-0 whitespace-nowrap">Public ID</TableHead>
              <TableHead>Description</TableHead>
              <TableHead className="w-0 whitespace-nowrap">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {apiKeys.length === 0 ? (
              <TableEmptyState message="No API keys found" />
            ) : (
              apiKeys.map((apiKey) => (
                <ApiKeyRow key={apiKey.public_id} apiKey={apiKey} />
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </TooltipProvider>
  );
}
