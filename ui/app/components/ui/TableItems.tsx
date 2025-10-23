import { Link } from "react-router";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { formatDate } from "~/utils/date";
import { useFunctionConfig } from "~/context/config";
import { AlertDialog } from "~/components/ui/AlertDialog";

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

interface TableItemTimeProps {
  timestamp: string | Date;
}

function TableItemTime({ timestamp }: TableItemTimeProps) {
  return (
    <span className="whitespace-nowrap">{formatDate(new Date(timestamp))}</span>
  );
}

interface TableItemTextProps {
  text: string | null;
}

function TableItemText({ text }: TableItemTextProps) {
  if (text === null) {
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

  const baseClasses =
    "flex items-center text-sm text-fg-primary gap-2 rounded-md font-mono";

  const content = (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`${functionIconConfig.iconBg} rounded-sm p-0.5`}>
            {functionIconConfig.icon}
          </div>
        </TooltipTrigger>
        <TooltipContent
          className="border-border bg-bg-secondary text-fg-primary border shadow-lg"
          sideOffset={5}
        >
          {functionIconConfig.label}
        </TooltipContent>
      </Tooltip>

      <span className="text-fg-primary inline-block truncate transition-colors duration-300 group-hover:text-gray-500">
        {functionName}
      </span>
    </>
  );

  if (link) {
    if (functionConfig) {
      return (
        <Link to={link} className={`${baseClasses} group cursor-pointer`}>
          {content}
        </Link>
      );
    } else {
      return (
        <AlertDialog
          message="This function is not present in your configuration file."
          trigger={
            <div className={`${baseClasses} cursor-default`}>{content}</div>
          }
        />
      );
    }
  }

  return <div className={`${baseClasses} cursor-default`}>{content}</div>;
}

export { TableItemFunction, TableItemShortUuid, TableItemText, TableItemTime };
