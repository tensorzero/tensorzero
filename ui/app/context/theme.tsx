import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export enum Theme {
  Light = "light",
  Dark = "dark",
  System = "system",
}

const STORAGE_KEY = "tensorzero-theme";

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: Theme.Light | Theme.Dark;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getSystemTheme(): Theme.Light | Theme.Dark {
  if (typeof window === "undefined") return Theme.Light;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? Theme.Dark
    : Theme.Light;
}

function getStoredTheme(): Theme {
  if (typeof window === "undefined") return Theme.System;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (
    stored === Theme.Light ||
    stored === Theme.Dark ||
    stored === Theme.System
  )
    return stored;
  return Theme.System;
}

function applyTheme(resolved: Theme.Light | Theme.Dark) {
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === Theme.Dark);
  root.style.colorScheme = resolved;
}

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);
  const [systemTheme, setSystemTheme] = useState<Theme.Light | Theme.Dark>(
    getSystemTheme,
  );

  const resolvedTheme =
    theme === Theme.System ? systemTheme : (theme as Theme.Light | Theme.Dark);

  // Listen for system preference changes
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? Theme.Dark : Theme.Light);
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Apply theme to DOM whenever resolved theme changes
  useEffect(() => {
    applyTheme(resolvedTheme);
  }, [resolvedTheme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    localStorage.setItem(STORAGE_KEY, next);
  }, []);

  const value = useMemo(
    () => ({ theme, resolvedTheme, setTheme }),
    [theme, resolvedTheme, setTheme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

/**
 * Inline script to prevent FOUC by applying the theme class before React hydrates.
 * Rendered in the <head> of root.tsx as a raw <script> tag.
 */
export const themeInitScript = `
(function(){
  try {
    var t = localStorage.getItem("${STORAGE_KEY}");
    var dark = t === "dark" || (t !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    if (dark) {
      document.documentElement.classList.add("dark");
      document.documentElement.style.colorScheme = "dark";
    }
  } catch(e) {}
})();
`;
