import { PlayIcon } from "lucide-react";
import { Button } from "~/components/ui/button";

const RunButton: React.FC<
  React.PropsWithChildren<{
    onRun?: () => void;
  }>
> = ({ onRun, children }) => {
  return (
    <Button
      className="rounded-full bg-gradient-to-br from-green-600 to-green-500 transition duration-150 hover:shadow-lg active:from-green-700 active:to-green-600"
      onClick={onRun}
    >
      {children ?? "Run"}
      <PlayIcon className="h-3 w-3" />
    </Button>
  );
};

export default RunButton;
