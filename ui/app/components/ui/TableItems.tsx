import { Link } from "react-router";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "./tooltip";
import { formatDate } from "~/utils/date";

interface TableItemShortUuidProps {
  id: string | null;
  link?: string;
}

function TableItemShortUuid({ id, link }: TableItemShortUuidProps) {
  if (id === null) {
    return <span className="text-fg-muted">—</span>;
  }

  const content = (
    <span
      className="inline-block max-w-[80px] overflow-hidden align-middle font-mono whitespace-nowrap"
      style={{ direction: "rtl", textOverflow: "ellipsis" }}
    >
      {id}
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
