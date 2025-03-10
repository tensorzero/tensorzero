import { useState } from "react";

type AlertDialogProps = {
  message: string;
  trigger: React.ReactNode;
};

export function AlertDialog({ message, trigger }: AlertDialogProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleTriggerClick = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsOpen(true);
    // Auto-close after 3 seconds
    setTimeout(() => setIsOpen(false), 2000);
  };

  return (
    <>
      <span onClick={handleTriggerClick} style={{ cursor: "pointer" }}>
        {trigger}
      </span>

      {isOpen && (
        <div
          style={{
            position: "fixed",
            top: "20px",
            right: "20px",
            background: "#f44336",
            color: "white",
            padding: "12px 24px",
            borderRadius: "4px",
            boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
            zIndex: 1000,
          }}
        >
          {message}
        </div>
      )}
    </>
  );
}
