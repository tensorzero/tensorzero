import { useLocation } from "react-router";

export function useActivePath() {
  const location = useLocation();

  return (path: string) => {
    if (path === "/") {
      return location.pathname === "/";
    }
    return location.pathname.startsWith(path);
  };
}
