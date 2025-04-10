import * as React from "react";
import { Link } from "react-router";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "./tooltip";

interface TableItemIdProps extends React.HTMLAttributes<HTMLSpanElement> {
  id: string;
  link?: string;
}

const TableItemId = React.forwardRef<HTMLSpanElement, TableItemIdProps>(
  ({ id, link, ...props }, ref) => {
    const content = (
      <span ref={ref} className="font-mono whitespace-nowrap" {...props}>
        {id.length > 5 ? `…${id.slice(-8)}` : id}
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

export { TableItemId };
