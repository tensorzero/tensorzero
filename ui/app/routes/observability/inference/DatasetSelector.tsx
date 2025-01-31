import React, { useState } from 'react';
import { Plus, Check, Database } from 'lucide-react';
import { Button } from '~/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '~/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '~/components/ui/popover';
import { Badge } from '~/components/ui/badge';

const DatasetSelector = () => {
  const [open, setOpen] = useState(false);
  
  // Mock data - replace with your API data
  const recentDatasets = [
    { id: '1', name: 'Evaluation Set A', count: 156 },
    { id: '2', name: 'Training Data 2024', count: 1205 },
  ];
  
  const allDatasets = [
    ...recentDatasets,
    { id: '3', name: 'Edge Cases', count: 45 },
    { id: '4', name: 'Benchmark Suite', count: 89 },
  ];

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="w-48 justify-between">
          <Database className="mr-2 h-4 w-4" />
          Add to Dataset
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-96 p-0" align="start">
        <Command>
          <CommandInput placeholder="Search datasets..." />
          <CommandList>
            <CommandEmpty className="py-2 text-sm text-center">
              No datasets found
            </CommandEmpty>
            
            <CommandGroup heading="Recent">
              {recentDatasets.map((dataset) => (
                <CommandItem 
                  key={dataset.id}
                  onSelect={() => {
                    // Handle selection
                    setOpen(false);
                  }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4 text-gray-500" />
                    {dataset.name}
                  </div>
                  <Badge variant="secondary" className="ml-2">
                    {dataset.count}
                  </Badge>
                </CommandItem>
              ))}
            </CommandGroup>

            <CommandSeparator />
            
            <CommandGroup heading="All Datasets">
              {allDatasets.map((dataset) => (
                <CommandItem 
                  key={dataset.id}
                  onSelect={() => {
                    // Handle selection
                    setOpen(false);
                  }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4 text-gray-500" />
                    {dataset.name}
                  </div>
                  <Badge variant="secondary" className="ml-2">
                    {dataset.count}
                  </Badge>
                </CommandItem>
              ))}
            </CommandGroup>

            <CommandSeparator />
            
            <CommandGroup>
              <CommandItem
                onSelect={() => {
                  // Handle new dataset creation
                  setOpen(false);
                }}
                className="flex items-center gap-2 text-blue-600"
              >
                <Plus className="h-4 w-4" />
                Create New Dataset
              </CommandItem>
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetSelector;