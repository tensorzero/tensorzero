import { useNavigate } from "react-router";
import { z } from "zod";
import { Button } from "~/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "~/components/ui/form";
import { Input } from "~/components/ui/input";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

const formSchema = z.object({
  episodeId: z.string().uuid("Please enter a valid UUID"),
});

type FormValues = z.infer<typeof formSchema>;

export default function EpisodeSearchBar() {
  const navigate = useNavigate();
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      episodeId: "",
    },
    mode: "onChange",
  });

  const onSubmit = (data: FormValues) => {
    navigate(`/observability/episodes/${data.episodeId}`);
    form.reset();
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="mb-4">
        <div className="flex gap-2">
          <FormField
            control={form.control}
            name="episodeId"
            render={({ field }) => (
              <FormItem className="flex-grow">
                <FormControl>
                  <Input
                    placeholder="00000000-0000-0000-0000-000000000000"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" disabled={!form.formState.isValid}>
            Go to Episode
          </Button>
        </div>
      </form>
    </Form>
  );
}
