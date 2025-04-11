import { Link } from "react-router";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "./tooltip";
import { formatDate } from "~/utils/date";

interface TableItemShortUuidProps {
  id: string;
  link?: string;
}

function TableItemShortUuid({ id, link }: TableItemShortUuidProps) {
  const content = (
    <span className="font-mono whitespace-nowrap">
      {id.length > 8 ? `…${id.slice(-8)}` : id}
    </span>
  );

  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          {link ? (
            <Link
              to={link}
              aria-label={id}
              className="block no-underline transition-colors duration-300 hover:text-gray-500"
            >
              {content}
            </Link>
          ) : (
            content
          )}
        </TooltipTrigger>
        <TooltipContent
          className="border-border bg-bg-secondary text-fg-primary border font-mono shadow-lg"
          sideOffset={5}
        >
          {id}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface TableItemTimeProps {
  timestamp: string | Date;
}

function TableItemTime({ timestamp }: TableItemTimeProps) {
  return (
    <span className="whitespace-nowrap">{formatDate(new Date(timestamp))}</span>
  );
}

export { TableItemShortUuid, TableItemTime };
