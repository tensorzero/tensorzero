"use client";

import {
  DndContext,
  type DragEndEvent,
  rectIntersection,
  useDraggable,
  useDroppable,
} from "@dnd-kit/core";
import { restrictToVerticalAxis } from "@dnd-kit/modifiers";
import type { ReactNode } from "react";
import { cn } from "~/utils/common";

export type { DragEndEvent } from "@dnd-kit/core";

type Status = {
  id: string;
  name: string;
  color: string;
};

type Feature = {
  id: string;
  name: string;
  startAt: Date;
  endAt: Date;
  status: Status;
};

export type ListItemsProps = {
  children: ReactNode;
  className?: string;
};

export const ListItems = ({ children, className }: ListItemsProps) => (
  <div className={cn("flex flex-1 flex-col gap-2 p-3", className)}>
    {children}
  </div>
);

export type ListHeaderProps =
  | {
      children: ReactNode;
    }
  | {
      name: Status["name"];
      color: Status["color"];
      className?: string;
    };

export const ListHeader = (props: ListHeaderProps) =>
  "children" in props ? (
    props.children
  ) : (
    <div
      className={cn(
        "bg-foreground/5 flex shrink-0 items-center gap-2 p-3",
        props.className,
      )}
    >
      <div
        className="h-2 w-2 rounded-full"
        style={{ backgroundColor: props.color }}
      />
      <p className="m-0 text-sm font-semibold">{props.name}</p>
    </div>
  );

export type ListGroupProps = {
  id: Status["id"];
  children: ReactNode;
  className?: string;
};

export const ListGroup = ({ id, children, className }: ListGroupProps) => {
  const { setNodeRef, isOver } = useDroppable({ id });

  return (
    <div
      className={cn(
        "bg-secondary transition-colors",
        isOver && "bg-foreground/10",
        className,
      )}
      ref={setNodeRef}
    >
      {children}
    </div>
  );
};

export type ListItemProps = Pick<Feature, "id" | "name"> & {
  readonly index: number;
  readonly parent: string;
  readonly children?: ReactNode;
  readonly className?: string;
};

export const ListItem = ({
  id,
  name,
  index,
  parent,
  children,
  className,
}: ListItemProps) => {
  const { attributes, listeners, setNodeRef, transform, isDragging } =
    useDraggable({
      id,
      data: { index, parent },
    });

  return (
    <div
      className={cn(
        "bg-background flex cursor-grab items-center gap-2 rounded-md border p-2 shadow-sm",
        isDragging && "cursor-grabbing",
        className,
      )}
      style={{
        transform: transform
          ? `translateX(${transform.x}px) translateY(${transform.y}px)`
          : "none",
      }}
      {...listeners}
      {...attributes}
      ref={setNodeRef}
    >
      {children ?? <p className="m-0 text-sm font-medium">{name}</p>}
    </div>
  );
};

export type ListProviderProps = {
  children: ReactNode;
  onDragEnd: (event: DragEndEvent) => void;
  className?: string;
};

export const ListProvider = ({
  children,
  onDragEnd,
  className,
}: ListProviderProps) => (
  <DndContext
    collisionDetection={rectIntersection}
    modifiers={[restrictToVerticalAxis]}
    onDragEnd={onDragEnd}
  >
    <div className={cn("flex size-full flex-col", className)}>{children}</div>
  </DndContext>
);
