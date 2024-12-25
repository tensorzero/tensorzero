'use client'
import * as React from "react"
import { ChevronRight, ChevronLeft, FileText, LineChart, Binary } from 'lucide-react'

import { 
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
  SidebarTrigger,
  SidebarFooter,
  useSidebar,
} from "~/components/ui/sidebar"
import { cn } from "~/utils/common"
import { useLocation } from "react-router"

const navigation = [
  {
    title: "Observability",
    icon: LineChart,
    items: [
      {
        title: "Inferences",
        url: "#",
        description: "Monitor and analyze model inferences",
      },
      {
        title: "Episodes",
        url: "#",
        description: "Track interaction episodes",
      },
    ],
  },
  {
    title: "Optimization",
    icon: Binary,
    items: [
      {
        title: "Fine-tuning",
        url: "/optimization/fine-tuning",
        description: "Optimize model performance",
      },
    ],
  },
]

const bottomNavigation = [
  {
    title: "Documentation",
    icon: FileText,
    url: "https://www.tensorzero.com/docs",
  },
]

export function AppSidebar({ className, ...props }: React.ComponentProps<typeof Sidebar>) {
  const { state } = useSidebar()
  const location = useLocation()
  
  const isActive = (url: string) => location.pathname === url

  return (
    <Sidebar
      collapsible="icon" 
      className={cn(
        "h-screen transition-all duration-200",
        "border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
        className
      )} 
      {...props}
    >
      <div className={`flex flex-col h-full  ${
        state === 'collapsed' ? "mt-1.5" : "justify-between"
      }`}>
        <SidebarHeader className="border-b h-14 flex items-center px-3">
          <SidebarMenu className="">
            <SidebarMenuItem className="">
              <SidebarMenuButton size="lg" className="" asChild>
                <a href="/" className="flex opacity-95 hover:opacity-100 hover:text-black  mr-2 items-center gap-3">
                  <div className="flex aspect-square w-8 items-center justify-center rounded-lg bg-primary/5">
                    <img 
                      src="https://www.tensorzero.com/favicon.svg" 
                      alt="TensorZero logo" 
                      className={`w-6 h-6 `}
                    />
                  </div>
                  {state === 'expanded' && (
                    <span className="font-semibold text-gray-700 text-lg">TensorZero</span>
                  )}
                </a>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>

        <SidebarContent className="flex-1 justify-between px-3 py-4">
            <div>

           
          {navigation.map((section) => (
            <SidebarMenu key={section.title}>
              <SidebarMenuItem>
                <SidebarMenuButton asChild className="px-2">
                  <div className="flex items-center gap-2 h-8">
                  {state === 'expanded' && (
                    <>
                    {section.icon && (
                      <section.icon className="h-4 w-4 text-muted-foreground shrink-0" />
                    )}
                   
                      <span className="font-medium text-sm text-muted-foreground">{section.title}</span>
                      </>
                    )}
                  </div>
                </SidebarMenuButton>
                {section.items?.length && (
                  <SidebarMenuSub>
                    {section.items.map((item) => (
                      <SidebarMenuSubItem key={item.title}>
                        <SidebarMenuSubButton
                          asChild
                          className={cn(
                            "px-2 py-1.5 transition-colors hover:bg-accent rounded-md",
                            isActive(item.url) && "bg-accent",
                            state === 'collapsed' ? "justify-center" : "ml-6"
                          )}
                        >
                          <a href={item.url}>
                            {state === 'collapsed' ? (
                              <span className="w-2 h-2 rounded-full bg-current" />
                            ) : (
                              <span className="font-medium text-sm">{item.title}</span>
                            )}
                          </a>
                        </SidebarMenuSubButton>
                      </SidebarMenuSubItem>
                    ))}
                  </SidebarMenuSub>
                )}
              </SidebarMenuItem>
            </SidebarMenu>
          ))}
           </div>
          
          <div className="">
            {bottomNavigation.map((item) => (
              <SidebarMenu key={item.title}>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className={cn(
                    "px-2 py-1.5",
                    // state === 'collapsed' && "justify-center"
                  )}>
                    <a href={item.url} target="_blank" className="flex items-center gap-2">
                      <item.icon className="h-4 w-4 text-muted-foreground shrink-0" />
                      {state === 'expanded' && (
                        <span className="text-sm">{item.title}</span>
                      )}
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            ))}
          </div>
        </SidebarContent>

        <SidebarFooter className="border-t">
          

          <div className={cn("flex p-2.5", state === 'collapsed' ? "justify-center" : "justify-end")}>
            <SidebarTrigger>
              <button className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-md hover:bg-primary/90 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                {state === 'expanded' ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                <span className="sr-only">
                  {state === 'expanded' ? 'Collapse' : 'Expand'} sidebar
                </span>
              </button>
            </SidebarTrigger>
          </div>
        </SidebarFooter>
        <SidebarRail />
      </div>
    </Sidebar>
  )
}