import { Moon, Sun, Monitor } from "lucide-react";
import { Theme, useTheme } from "~/context/theme";
import {
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "~/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";

const THEME_OPTIONS = [
  { value: Theme.Light, label: "Light", icon: Sun },
  { value: Theme.Dark, label: "Dark", icon: Moon },
  { value: Theme.System, label: "System", icon: Monitor },
] as const;

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const { state } = useSidebar();

  const ActiveIcon = resolvedTheme === Theme.Dark ? Moon : Sun;

  const currentLabel = THEME_OPTIONS.find((o) => o.value === theme)?.label;

  return (
    <SidebarMenuItem className="list-none">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <SidebarMenuButton
            aria-label="Toggle theme"
            className="w-auto cursor-pointer"
            tooltip={state === "collapsed" ? "Toggle theme" : undefined}
          >
            <ActiveIcon className="h-4 w-4" />
            {state === "expanded" && (
              <span className="whitespace-nowrap transition-opacity duration-200 group-data-[collapsible=icon]:opacity-0">
                {currentLabel}
              </span>
            )}
          </SidebarMenuButton>
        </DropdownMenuTrigger>
        <DropdownMenuContent side="top" align="start">
          {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
            <DropdownMenuItem
              key={value}
              onClick={() => setTheme(value)}
              className={`cursor-pointer ${theme === value ? "bg-accent" : ""}`}
            >
              <Icon className="mr-2 h-4 w-4" />
              {label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  );
}
