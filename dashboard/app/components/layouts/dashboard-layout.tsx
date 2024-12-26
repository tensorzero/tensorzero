import { Outlet } from "react-router";

import { SidebarProvider } from "~/components/ui/sidebar";
import { AppSidebar } from "~/components/ui/sidebar/app.sidebar";

export default function DashboardLayout() {
  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <AppSidebar />
        <main className="flex-1 min-h-screen w-full overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </SidebarProvider>
  );
}