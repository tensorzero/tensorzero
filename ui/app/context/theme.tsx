"use client";

import { createContext, use, useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

const THEME_COOKIE_NAME = "theme";
const THEME_COOKIE_MAX_AGE = 60 * 60 * 24 * 365; // 1 year

const ThemeContext = createContext<{
  theme: Theme;
  setTheme: (theme: Theme) => void;
}>({ theme: "system", setTheme: () => {} });

export function useTheme() {
  return use(ThemeContext);
}

function applyTheme(theme: Theme) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.classList.toggle("dark", isDark);
}

export function ThemeProvider({
  children,
  defaultTheme = "system",
}: {
  children: React.ReactNode;
  defaultTheme?: Theme;
}) {
  const [theme, setThemeState] = useState<Theme>(defaultTheme);

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
    document.cookie = `${THEME_COOKIE_NAME}=${newTheme}; path=/; max-age=${THEME_COOKIE_MAX_AGE}`;
    applyTheme(newTheme);
  }, []);

  // Apply theme on mount and listen for system preference changes
  useEffect(() => {
    applyTheme(theme);

    if (theme !== "system") return;

    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("system");
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [theme]);

  return <ThemeContext value={{ theme, setTheme }}>{children}</ThemeContext>;
}
