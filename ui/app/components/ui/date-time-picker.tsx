import * as React from "react";
import { CalendarIcon } from "lucide-react";
import { formatDate } from "~/utils/date";

import { Button } from "~/components/ui/button";
import { Calendar } from "~/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { cn } from "~/utils/common";

interface DateTimePickerProps {
  value?: Date;
  onChange?: (date: Date | undefined) => void;
  placeholder?: string;
  disabled?: boolean;
  minDate?: Date;
  id?: string;
  className?: string;
  "aria-describedby"?: string;
  "aria-invalid"?: boolean;
}

function toHour12(date: Date): number {
  return date.getHours() % 12 || 12;
}

function toAmPm(date: Date): "AM" | "PM" {
  return date.getHours() >= 12 ? "PM" : "AM";
}

function toHours24(hour12: number, ampm: "AM" | "PM"): number {
  if (ampm === "AM") return hour12 === 12 ? 0 : hour12;
  return hour12 === 12 ? 12 : hour12 + 12;
}

function startOfDay(date: Date): Date {
  const startOfDate = new Date(date);
  startOfDate.setHours(0, 0, 0, 0);
  return startOfDate;
}

const HOURS = Array.from({ length: 12 }, (_, i) => i + 1);
const MINUTES = Array.from({ length: 60 }, (_, i) => i);

export function DateTimePicker({
  value,
  onChange,
  placeholder = "Pick a date and time",
  disabled,
  minDate,
  id,
  className,
  "aria-describedby": ariaDescribedBy,
  "aria-invalid": ariaInvalid,
}: DateTimePickerProps) {
  const [open, setOpen] = React.useState(false);
  const [hour, setHour] = React.useState<number>(value ? toHour12(value) : 12);
  const [minute, setMinute] = React.useState<number>(
    value ? value.getMinutes() : 0,
  );
  const [ampm, setAmPm] = React.useState<"AM" | "PM">(
    value ? toAmPm(value) : "AM",
  );
  const minCalendarDate = minDate ? startOfDay(minDate) : undefined;

  const applyTime = (h: number, m: number, ap: "AM" | "PM", base?: Date) => {
    const day = base ?? value;
    if (!day) return;
    const combined = new Date(day);
    combined.setHours(toHours24(h, ap), m, 0, 0);
    onChange?.(combined);
  };

  const handleDaySelect = (day: Date | undefined) => {
    if (!day) {
      onChange?.(undefined);
      return;
    }
    applyTime(hour, minute, ampm, day);
  };

  const handleHourChange = (val: string) => {
    const h = Number(val);
    setHour(h);
    applyTime(h, minute, ampm);
  };

  const handleMinuteChange = (val: string) => {
    const m = Number(val);
    setMinute(m);
    applyTime(hour, m, ampm);
  };

  const handleAmPmChange = (val: string) => {
    const ap = val as "AM" | "PM";
    setAmPm(ap);
    applyTime(hour, minute, ap);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          id={id}
          variant="outline"
          disabled={disabled}
          aria-describedby={ariaDescribedBy}
          aria-invalid={ariaInvalid}
          className={cn(
            "w-full justify-start text-left font-normal",
            !value && "text-muted-foreground",
            ariaInvalid &&
              "border-destructive focus-visible:border-destructive focus-visible:ring-destructive/20",
            className,
          )}
        >
          <CalendarIcon className="mr-2 size-4" />
          {value ? formatDate(value) : <span>{placeholder}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={value}
          onSelect={handleDaySelect}
          disabled={
            minCalendarDate ? (day) => day < minCalendarDate : undefined
          }
          autoFocus
        />
        <div className="border-t p-3 flex flex-col gap-3 items-center">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Select value={String(hour)} onValueChange={handleHourChange}>
                <SelectTrigger className="h-8 w-[4.25rem] px-2 [&>span]:flex-1 [&>span]:text-center [&>span]:tabular-nums">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {HOURS.map((h) => (
                    <SelectItem key={h} value={String(h)}>
                      {String(h).padStart(2, "0")}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-muted-foreground text-sm font-medium">
                :
              </span>
              <Select value={String(minute)} onValueChange={handleMinuteChange}>
                <SelectTrigger className="h-8 w-[4.25rem] px-2 [&>span]:flex-1 [&>span]:text-center [&>span]:tabular-nums">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MINUTES.map((m) => (
                    <SelectItem key={m} value={String(m)}>
                      {String(m).padStart(2, "0")}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={ampm} onValueChange={handleAmPmChange}>
                <SelectTrigger className="h-8 w-[5rem] px-2 [&>span]:flex-1 [&>span]:text-center">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="AM">AM</SelectItem>
                  <SelectItem value="PM">PM</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <Button
            type="button"
            size="sm"
            className="w-full"
            onClick={() => setOpen(false)}
          >
            Done
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
