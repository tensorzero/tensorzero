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
  inferenceId: z.string().uuid("Please enter a valid UUID"),
});

type FormValues = z.infer<typeof formSchema>;

export default function InferenceSearchBar() {
  const navigate = useNavigate();
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      inferenceId: "",
    },
    mode: "onChange",
  });

  const onSubmit = (data: FormValues) => {
    navigate(`/observability/inferences/${data.inferenceId}`);
    form.reset();
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="mb-4">
        <div className="flex gap-2">
          <FormField
            control={form.control}
            name="inferenceId"
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
            Go to Inference
          </Button>
        </div>
      </form>
    </Form>
  );
}
