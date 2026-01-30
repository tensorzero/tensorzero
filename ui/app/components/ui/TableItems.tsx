import { Link } from "react-router";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { formatDateOnly, formatTimeOnly } from "~/utils/date";
import { useFunctionConfig } from "~/context/config";
import { useToast } from "~/hooks/use-toast";

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
      className="inline-block max-w-[80px] overflow-hidden align-middle font-mono text-ellipsis whitespace-nowrap"
      dir="rtl"
    >
      {id}
    </span>
  );

  return (
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
  );
}

function DotSeparator() {
  return (
    <span
      className="bg-fg-muted inline-block h-0.5 w-0.5 rounded-full"
      aria-hidden="true"
    />
  );
}

interface TableItemTimeProps {
  timestamp: string | Date;
}

function TableItemTime({ timestamp }: TableItemTimeProps) {
  const date = new Date(timestamp);
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      {formatDateOnly(date)}
      <DotSeparator />
      {formatTimeOnly(date)}
    </span>
  );
}

interface TableItemTextProps {
  text?: string;
}

function TableItemText({ text }: TableItemTextProps) {
  if (text === undefined) {
    return <span className="text-fg-muted">—</span>;
  } else {
    return <span className="whitespace-nowrap">{text}</span>;
  }
}

interface TableItemFunctionProps {
  functionName: string;
  functionType: string;
  link?: string;
}

function TableItemFunction({
  functionName,
  functionType,
  link,
}: TableItemFunctionProps) {
  const functionIconConfig = getFunctionTypeIcon(functionType);
  const functionConfig = useFunctionConfig(functionName);
  const { toast } = useToast();

  const baseClasses =
    "flex items-center text-sm text-fg-primary gap-2 rounded-md font-mono group";

  const content = (
    <>
      <span
        className={`${functionIconConfig.iconBg} rounded-sm p-0.5`}
        aria-hidden
      >
        {functionIconConfig.icon}
      </span>
      <span className="text-fg-primary inline-block truncate transition-colors duration-300 group-hover:text-gray-500">
        {functionName}
      </span>
    </>
  );

  if (link) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          {functionConfig ? (
            <Link to={link} className={`${baseClasses} cursor-pointer`}>
              {content}
            </Link>
          ) : (
            <button
              type="button"
              className={`${baseClasses} w-full cursor-default`}
              onClick={() => {
                toast.error({
                  description:
                    "This function is not present in your configuration file.",
                });
              }}
            >
              {content}
            </button>
          )}
        </TooltipTrigger>
        <TooltipContent align="start">
          {functionIconConfig.label}
        </TooltipContent>
      </Tooltip>
    );
  }

  return <div className={`${baseClasses} cursor-default`}>{content}</div>;
}

export {
  DotSeparator,
  TableItemFunction,
  TableItemShortUuid,
  TableItemText,
  TableItemTime,
};
