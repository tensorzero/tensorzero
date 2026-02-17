import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import { HoverCard } from "radix-ui";
import type { ResolvedObject } from "~/types/tensorzero";
import { EntityPreviewType, useEntityPreview } from "~/hooks/useEntityPreview";
import { getRelativeTimeString, getTimestampTooltipData } from "~/utils/date";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useFunctionConfig } from "~/context/config";
import { toResolvedObjectUrl } from "~/utils/urls";

interface UuidHoverCardProps {
  uuid: string;
  obj: ResolvedObject;
  children: React.ReactNode;
}

export function UuidHoverCard({ uuid, obj, children }: UuidHoverCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const url = toResolvedObjectUrl(uuid, obj);

  if (!url) {
    return <>{children}</>;
  }

  return (
    <HoverCard.Root openDelay={300} closeDelay={200} onOpenChange={setIsOpen}>
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={4}
          className={cn(
            "bg-popover text-popover-foreground z-50 rounded-md border p-3 shadow-md",
            obj.type === "inference"
              ? "w-80"
              : obj.type === "episode"
                ? "w-44"
                : "w-56",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
          )}
        >
          <HoverCardContent uuid={uuid} obj={obj} isOpen={isOpen} />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}

interface HoverCardContentProps {
  uuid: string;
  obj: ResolvedObject;
  isOpen: boolean;
}

function HoverCardContent({ uuid, obj, isOpen }: HoverCardContentProps) {
  switch (obj.type) {
    case "inference":
      return <InferenceContent uuid={uuid} obj={obj} isOpen={isOpen} />;
    case "episode":
      return <EpisodeContent uuid={uuid} obj={obj} isOpen={isOpen} />;
    case "chat_datapoint":
    case "json_datapoint":
      return <DatapointContent uuid={uuid} obj={obj} />;
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default: {
      const _exhaustiveCheck: never = obj;
      return _exhaustiveCheck;
    }
  }
}

interface InferencePreview {
  timestamp: string;
}

interface InferenceContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "inference" }>;
  isOpen: boolean;
}

function InferenceContent({ uuid, obj, isOpen }: InferenceContentProps) {
  const { data, isLoading } = useEntityPreview<InferencePreview>({
    type: EntityPreviewType.Inference,
    id: uuid,
    enabled: isOpen,
  });
  const functionConfig = useFunctionConfig(obj.function_name);
  const variantType =
    functionConfig?.variants[obj.variant_name]?.inner.type ?? null;

  return (
    <div className="flex flex-col gap-2">
      <TypeBadgeLink uuid={uuid} obj={obj}>
        Inference
      </TypeBadgeLink>
      <FunctionItem
        functionName={obj.function_name}
        functionType={obj.function_type}
      />
      <InfoItem
        label="Variant"
        value={obj.variant_name}
        secondaryValue={variantType}
      />
      <Timestamp data={data} isLoading={isLoading} />
    </div>
  );
}

interface EpisodePreview {
  inference_count: number;
}

interface EpisodeContentProps {
  uuid: string;
  obj: ResolvedObject;
  isOpen: boolean;
}

function EpisodeContent({ uuid, obj, isOpen }: EpisodeContentProps) {
  const { data, isLoading } = useEntityPreview<EpisodePreview>({
    type: EntityPreviewType.Episode,
    id: uuid,
    enabled: isOpen,
  });

  return (
    <div className="flex flex-col gap-2">
      <TypeBadgeLink uuid={uuid} obj={obj}>
        Episode
      </TypeBadgeLink>
      <InfoItem
        label="Inferences"
        value={data ? String(data.inference_count) : null}
        isLoading={isLoading}
      />
    </div>
  );
}

interface DatapointContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "chat_datapoint" | "json_datapoint" }>;
}

function DatapointContent({ uuid, obj }: DatapointContentProps) {
  const typeLabel =
    obj.type === "chat_datapoint" ? "Chat Datapoint" : "JSON Datapoint";

  return (
    <div className="flex flex-col gap-2">
      <TypeBadgeLink uuid={uuid} obj={obj}>
        {typeLabel}
      </TypeBadgeLink>
      <InfoItem label="Dataset" value={obj.dataset_name} />
      <InfoItem label="Function" value={obj.function_name} />
    </div>
  );
}

interface TypeBadgeLinkProps {
  uuid: string;
  obj: ResolvedObject;
  children: React.ReactNode;
}

function TypeBadgeLink({ uuid, obj, children }: TypeBadgeLinkProps) {
  const url = toResolvedObjectUrl(uuid, obj);
  if (!url) return null;

  return (
    <Link
      to={url}
      className="text-muted-foreground hover:text-foreground inline-flex items-center text-xs transition-colors"
    >
      {children}
      <ChevronRight className="ml-0.5 h-3 w-3" />
    </Link>
  );
}

interface ItemProps {
  label: string;
  children: React.ReactNode;
}

function Item({ label, children }: ItemProps) {
  return (
    <div className="grid grid-cols-[4rem_1fr] items-center gap-2">
      <span className="text-muted-foreground text-xs">{label}</span>
      {children}
    </div>
  );
}

interface FunctionItemProps {
  functionName: string;
  functionType: string;
}

function FunctionItem({ functionName, functionType }: FunctionItemProps) {
  const iconConfig = getFunctionTypeIcon(functionType);
  return (
    <Item label="Function">
      <span
        className="text-foreground inline-flex min-w-0 items-center gap-1 font-mono text-xs"
        title={`${functionName} · ${functionType}`}
      >
        <span
          className={cn(
            "inline-flex shrink-0 items-center justify-center rounded p-0.5",
            iconConfig.iconBg,
          )}
        >
          {iconConfig.icon}
        </span>
        <span className="truncate">{functionName}</span>
        <span className="text-muted-foreground shrink-0">· {functionType}</span>
      </span>
    </Item>
  );
}

interface InfoItemProps {
  label: string;
  value?: string | null;
  secondaryValue?: string | null;
  isLoading?: boolean;
}

function InfoItem({ label, value, secondaryValue, isLoading }: InfoItemProps) {
  return (
    <Item label={label}>
      {value ? (
        <span
          className="text-foreground min-w-0 truncate font-mono text-xs"
          title={secondaryValue ? `${value} · ${secondaryValue}` : value}
        >
          {value}
          {secondaryValue && (
            <span className="text-muted-foreground"> · {secondaryValue}</span>
          )}
        </span>
      ) : isLoading ? (
        <span className="bg-muted h-4 w-20 animate-pulse rounded" />
      ) : null}
    </Item>
  );
}

interface TimestampProps {
  data: InferencePreview | null;
  isLoading: boolean;
}

function Timestamp({ data, isLoading }: TimestampProps) {
  if (data) {
    const date = new Date(data.timestamp);
    const { formattedDate, formattedTime } = getTimestampTooltipData(
      data.timestamp,
    );
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-muted-foreground w-fit cursor-default text-xs">
            {getRelativeTimeString(date)}
          </span>
        </TooltipTrigger>
        <TooltipContent className="border-border bg-bg-secondary text-fg-primary border shadow-lg">
          <div className="flex flex-col gap-1">
            <div>{formattedDate}</div>
            <div>{formattedTime}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    );
  }
  if (isLoading) {
    return <span className="bg-muted h-4 w-16 animate-pulse rounded" />;
  }
  return null;
}
