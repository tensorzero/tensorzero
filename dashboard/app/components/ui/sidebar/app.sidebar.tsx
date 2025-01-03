import * as React from "react";
import { ChevronRight, ChevronLeft, FileText } from "lucide-react";
import { useSidebar } from "~/components/ui/sidebar";
import { cn } from "~/utils/common";

import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
  SidebarTrigger,
  SidebarFooter,
} from "~/components/ui/sidebar";

const navigation = [
  {
    title: "Observability",
    items: [
      {
        title: "Inferences",
        url: "/observability/inferences",
      },
      {
        title: "Episodes",
        url: "#",
      },
    ],
  },
  {
    title: "Optimization",
    items: [
      {
        title: "Supervised Fine-Tuning",
        url: "/optimization/supervised-fine-tuning",
      },
    ],
  },
];

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { state } = useSidebar();

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="/" className="flex items-center gap-3">
                <div className="flex aspect-square size-8 items-center justify-center">
                  <img
                    src="https://www.tensorzero.com/favicon.svg"
                    alt="TensorZero logo"
                    className="size-full"
                  />
                </div>
                <span className="font-semibold">TensorZero</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent
        className={cn(
          "!overflow-y-auto !overflow-x-hidden transition-[width] duration-200",
        )}
      >
        {state === "expanded" &&
          navigation.map((section) => (
            <SidebarMenu key={section.title}>
              <SidebarMenuItem>
                <span className="block w-full px-2 py-1.5 text-sm font-medium text-sidebar-foreground">
                  {section.title}
                </span>
                {section.items?.length ? (
                  <SidebarMenuSub>
                    {section.items.map((item) => (
                      <SidebarMenuSubItem key={item.title}>
                        <SidebarMenuSubButton asChild>
                          <a href={item.url}>{item.title}</a>
                        </SidebarMenuSubButton>
                      </SidebarMenuSubItem>
                    ))}
                  </SidebarMenuSub>
                ) : null}
              </SidebarMenuItem>
            </SidebarMenu>
          ))}
      </SidebarContent>
      <SidebarFooter className="relative">
        <div
          className={cn(
            "flex items-center py-2",
            state === "collapsed" ? "justify-center" : "justify-between px-2",
          )}
        >
          {state === "expanded" && (
            <a
              href="https://www.tensorzero.com/docs"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open documentation in new tab"
            >
              <FileText className="h-4 w-4" />
              Documentation
            </a>
          )}
          <SidebarTrigger>
            <button className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-md hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring">
              {state === "expanded" ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              <span className="sr-only">
                {state === "expanded" ? "Collapse" : "Expand"} sidebar
              </span>
            </button>
          </SidebarTrigger>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
