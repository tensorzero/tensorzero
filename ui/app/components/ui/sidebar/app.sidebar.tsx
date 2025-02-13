import * as React from "react";
import { ChevronRight, ChevronLeft, FileText, ChartSpline, Layers, SquareFunction, View } from "lucide-react";
import { useSidebar } from "~/components/ui/sidebar";
import { cn } from "~/utils/common";

import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarRail,
  SidebarTrigger,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "~/components/ui/sidebar";

const navigation = [
  {
    title: "Observability",
    items: [
      {
        title: "Inferences",
        url: "/observability/inferences",
        icon: ChartSpline,
      },
      {
        title: "Episodes",
        url: "/observability/episodes",
        icon: Layers,
      },
      {
        title: "Functions",
        url: "/observability/functions",
        icon: SquareFunction,
      },
    ],
  },
  {
    title: "Optimization",
    items: [
      {
        title: "Supervised Fine-Tuning",
        url: "/optimization/supervised-fine-tuning",
        icon: View,
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
              <a href="/" className="flex items-center gap-2">
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
        {navigation.map((section) => (
          <SidebarGroup key={section.title}>
            {state === "expanded" && (
              <SidebarGroupLabel>
                {section.title}
              </SidebarGroupLabel>
            )}
            <SidebarGroupContent>
              {section.items?.map((item) => (
                <SidebarMenuItem key={item.title} className="list-none">
                  <SidebarMenuButton 
                    asChild
                    tooltip={state === "collapsed" ? item.title : undefined}
                  >
                    <a
                      href={item.url}
                      className="flex items-center gap-3 px-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <item.icon className="h-4 w-4" />
                      {state === "expanded" && <span>{item.title}</span>}
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarFooter className="relative">
        <SidebarTrigger className="flex justify-left">
          {state === "expanded" ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <span className="sr-only">
            {state === "expanded" ? "Collapse" : "Expand"} sidebar
          </span>
        </SidebarTrigger>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton 
              asChild
              tooltip={state === "collapsed" ? "Documentation" : undefined}
            >
              <a
                href="https://www.tensorzero.com/docs"
                className="flex items-center gap-3 px-2"
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <FileText className="h-4 w-4" />
                {state === "expanded" && <span>Documentation</span>}
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
