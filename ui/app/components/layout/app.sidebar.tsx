import * as React from "react";
import {
  ChevronRight,
  ChevronLeft,
  BookOpenText,
  ChartSpline,
  GalleryVerticalEnd,
  SquareFunction,
  View,
  Home,
  type LucideIcon,
} from "lucide-react";
import { useSidebar } from "~/components/ui/sidebar";
import { cn } from "~/utils/common";
import { useActivePath } from "~/hooks/use-active-path";
import { TensorZeroLogo } from "~/components/icons/Icons";
import { Link } from "react-router";

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

interface NavigationItem {
  title: string;
  url: string;
  icon: LucideIcon;
}

interface NavigationSection {
  title: string;
  items: NavigationItem[];
}

const navigation: NavigationSection[] = [
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
        icon: GalleryVerticalEnd,
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
  const isActivePath = useActivePath();

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader className="pb-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="text-black hover:bg-transparent focus-visible:bg-transparent active:bg-transparent"
            >
              <Link to="/" className="flex items-center gap-2">
                <TensorZeroLogo size={16} />
                {state === "expanded" && (
                  <span className="font-semibold">TensorZero</span>
                )}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent
        className={cn(
          "!overflow-y-auto !overflow-x-hidden transition-[width] duration-200",
        )}
      >
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenuItem className="list-none">
              <SidebarMenuButton
                asChild
                tooltip={state === "collapsed" ? "Dashboard" : undefined}
                isActive={isActivePath("/")}
              >
                <Link to="/" className="flex items-center gap-2">
                  <Home className="h-4 w-4" />
                  {state === "expanded" && <span>Dashboard</span>}
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarGroupContent>
        </SidebarGroup>
        {navigation.map((section) => (
          <SidebarGroup key={section.title}>
            {state === "expanded" && (
              <SidebarGroupLabel>{section.title}</SidebarGroupLabel>
            )}
            <SidebarGroupContent className="flex flex-col gap-1">
              {section.items?.map((item) => (
                <SidebarMenuItem key={item.title} className="list-none">
                  <SidebarMenuButton
                    asChild
                    tooltip={state === "collapsed" ? item.title : undefined}
                    isActive={isActivePath(item.url)}
                  >
                    <Link
                      to={item.url}
                      className="flex items-center gap-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <item.icon className="h-4 w-4" />
                      {state === "expanded" && <span>{item.title}</span>}
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
        <SidebarGroup>
          {state === "expanded" && <SidebarGroupLabel>Other</SidebarGroupLabel>}
          <SidebarGroupContent>
            <SidebarMenuItem className="list-none">
              <SidebarMenuButton
                asChild
                tooltip={state === "collapsed" ? "Documentation" : undefined}
              >
                <Link
                  to="https://www.tensorzero.com/docs"
                  className="flex items-center gap-2"
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  <BookOpenText className="h-4 w-4" />
                  {state === "expanded" && (
                    <span>
                      Documentation <sup>↗</sup>
                    </span>
                  )}
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="relative">
        <SidebarTrigger className="justify-left flex">
          {state === "expanded" ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <span className="sr-only">
            {state === "expanded" ? "Collapse sidebar" : "Expand sidebar"}
          </span>
        </SidebarTrigger>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
