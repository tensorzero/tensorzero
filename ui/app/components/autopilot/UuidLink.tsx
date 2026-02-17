import { useState } from "react";
import { Link } from "react-router";
import { HoverCard } from "radix-ui";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";
import { cn } from "~/utils/common";
import { UuidHoverCardContent } from "./UuidHoverCardContent";

function getUrlForResolvedObject(
  uuid: string,
  obj: ResolvedObject,
): string | null {
  switch (obj.type) {
    case "inference":
      return toInferenceUrl(uuid);
    case "episode":
      return toEpisodeUrl(uuid);
    case "chat_datapoint":
    case "json_datapoint":
      return toDatapointUrl(obj.dataset_name, uuid);
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

export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);
  const [isOpen, setIsOpen] = useState(false);

  const obj = data?.object_types.length === 1 ? data.object_types[0] : null;
  const url = obj ? getUrlForResolvedObject(uuid, obj) : null;

  if (!url || !obj) {
    return (
      <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs font-medium">
        {uuid}
      </code>
    );
  }

  return (
    <HoverCard.Root openDelay={300} closeDelay={200} onOpenChange={setIsOpen}>
      <HoverCard.Trigger asChild>
        <Link
          to={url}
          className="rounded bg-orange-50 px-1 py-0.5 font-mono text-xs text-orange-500 no-underline hover:underline"
        >
          {uuid}
        </Link>
      </HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={4}
          className={cn(
            "bg-popover text-popover-foreground z-50 rounded-md border p-3 shadow-md",
            obj.type === "episode" ? "w-56" : "w-80",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
          )}
        >
          <UuidHoverCardContent
            uuid={uuid}
            obj={obj}
            url={url}
            isOpen={isOpen}
          />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}
