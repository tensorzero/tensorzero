import { Link } from "react-router";
import { Button, ButtonIcon } from "~/components/ui/button";
import { Chat } from "~/components/icons/Icons";
import { useAutopilotAvailable } from "~/context/autopilot-available";

interface AskAutopilotButtonProps {
  message: string;
}

export function AskAutopilotButton({ message }: AskAutopilotButtonProps) {
  const autopilotAvailable = useAutopilotAvailable();

  if (!autopilotAvailable) {
    return null;
  }

  const params = new URLSearchParams({ message });

  return (
    <Button variant="outline" size="sm" asChild>
      <Link to={`/autopilot/sessions/new?${params.toString()}`}>
        <ButtonIcon as={Chat} variant="tertiary" />
        Ask Autopilot
      </Link>
    </Button>
  );
}
