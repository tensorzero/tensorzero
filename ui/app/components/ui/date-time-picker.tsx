import * as React from "react";
import { format } from "date-fns";
import { CalendarIcon } from "lucide-react";

import { Button } from "~/components/ui/button";
import { Calendar } from "~/components/ui/calendar";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { cn } from "~/utils/common";

interface DateTimePickerProps {
  value?: Date;
  onChange?: (date: Date | undefined) => void;
  placeholder?: string;
  disabled?: boolean;
  minDate?: Date;
  id?: string;
}

export function DateTimePicker({
  value,
  onChange,
  placeholder = "Pick a date and time",
  disabled,
  minDate,
  id,
}: DateTimePickerProps) {
  const [open, setOpen] = React.useState(false);
  const [timeValue, setTimeValue] = React.useState(
    value ? format(value, "HH:mm") : "00:00",
  );

  const handleDaySelect = (day: Date | undefined) => {
    if (!day) {
      onChange?.(undefined);
      return;
    }
    const [hours, minutes] = timeValue.split(":").map(Number);
    const combined = new Date(day);
    combined.setHours(hours, minutes, 0, 0);
    onChange?.(combined);
  };

  const handleTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = e.target.value;
    setTimeValue(time);
    if (!value) return;
    const [hours, minutes] = time.split(":").map(Number);
    const combined = new Date(value);
    combined.setHours(hours, minutes, 0, 0);
    onChange?.(combined);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          variant="outline"
          disabled={disabled}
          className={cn(
            "w-full justify-start text-left font-normal",
            !value && "text-muted-foreground",
          )}
        >
          <CalendarIcon className="mr-2 size-4" />
          {value ? format(value, "PPP HH:mm") : <span>{placeholder}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={value}
          onSelect={handleDaySelect}
          disabled={minDate ? (day) => day < minDate : undefined}
          initialFocus
        />
        <div className="border-t p-3">
          <div className="flex items-center gap-2">
            <Label htmlFor="time-input" className="text-sm">
              Time
            </Label>
            <Input
              id="time-input"
              type="time"
              value={timeValue}
              onChange={handleTimeChange}
              className="h-8 w-28"
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
