import * as React from "react";
import { Link } from "react-router";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "./tooltip";
import { formatDate } from "~/utils/date";

interface TableItemIdProps {
  id: string;
  link?: string;
}

const TableItemId = React.forwardRef<HTMLSpanElement, TableItemIdProps>(
  ({ id, link }, ref) => {
    const content = (
      <span ref={ref} className="font-mono whitespace-nowrap">
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
  },
);
TableItemId.displayName = "TableItemId";

interface TableItemTimeProps {
  timestamp: string | Date;
}

const TableItemTime = React.forwardRef<HTMLSpanElement, TableItemTimeProps>(
  ({ timestamp }, ref) => {
    return (
      <span ref={ref} className="whitespace-nowrap">
        {formatDate(new Date(timestamp))}
      </span>
    );
  },
);
TableItemTime.displayName = "TableItemTime";

export { TableItemId, TableItemTime };
