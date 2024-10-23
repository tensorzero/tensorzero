# Example: Improving LLM Chess Ability with Mixture-of-N Sampling

TensorZero was built to support inference strategies more sophisticated than just a single chat completion.
Today, we also support a mixture-of-n variant type which samples from several variants concurrently and uses another LLM call to combine the answers into a winner.

In this example, we'll show how you can drop in this experimental mixture-of-n variant type to spend additional compute budget for better performance on a challenging LLM benchmark.

## Background: Chess Puzzles

Chess puzzles are tactical challenges designed to test and improve a player's chess skills. They typically present a specific board position where the player must find the best move or sequence of moves to achieve a particular goal, such as checkmate, gaining material advantage, or forcing a draw. These puzzles are not only excellent training tools for chess players of all levels but also serve as an engaging way to assess an AI's ability to understand and apply chess strategies in complex situations.

We pulled a large dataset of [Lichess](https://lichess.org) chess puzzles from [Kaggle](https://www.kaggle.com/datasets/tianmin/lichess-chess-puzzle-dataset). After filtering for popular puzzles that were rated to be relatively easy (rating in [800, 1200]), we produced training and testing datasets with ~130k and ~15k puzzles respectively.

Each puzzle consists of a board position and a sequence of moves that solves it.
For example, here's a sample puzzle:

<p align="center">
<img src="img/puzzle.png" alt="Chess Puzzle">
<br>
<i>black to move</i>
</p>

The puzzle consists of a sequence of moves that are "forced" on the player: the player and their opponent would each be significantly disadvantaged if they missed their next move.
We can therefore evaluate an LLM by their exact match to the puzzle solution (or if they manage to achieve a checkmate).
Give this puzzle a try!

<details>
<summary><b>Solution</b></summary>

1. ... Qb1+
2. Re1 Qe1#

</details>

## Setup

### TensorZero

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and run the following command.
Docker Compose will launch the TensorZero Gateway and a test ClickHouse database.

```bash
docker compose up
```

### Python Environment

#### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv pip sync requirements.txt  # Install the dependencies
```

#### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```

## Running the Example

You can run the example in the `chess_puzzles.ipynb` notebook.
Make sure to install the dependencies in the `requirements.txt` file and set `CLICKHOUSE_URL=http://localhost:8123/tensorzero` in the shell your notebook will run in.
It should not require any changes to run and will automatically connect to the TensorZero Gateway you started.

The notebook will evaluate the performance of the default `gpt-4o-mini` variant on the test set of chess puzzles.
If you look at the `tensorzero.toml` file, you'll see that we've defined a mixture-of-n variant type for the `play_chess_board` function.
This means that we'll run 5 separate inference requests to the LLM, and use another LLM to combine the results into a single answer.
These are all instances of the `gpt-4o-mini` variant.
Without modifying the prompt or the model used, we can trade more tokens for a statistically significant improvement in performance (we saw ~10% relative improvement from 35% to 39% success rate with no prompt changes and further improvement to 41% with small variations to the prompt as in the section below).

Here are our results:

<p align="center">
  <img src="img/variant_success_rates.png" alt="Results">
</p>

## Modifying the Prompt

You might want to try diverse instructions for each LLM call.
We've included several other prompt templates in the `config/functions/play_chess_board/chess_*` directories.
We also can modify the `variant_name` variable to try out different mixture-of-n variants.
You can mix and match these in the `tensorzero.toml` file to try out different configurations.
We saw the diverse variant squeeze a few extra percentage points of performance from the same model and compute budget.
See for yourself if you can get better performance than our `gpt-4o-mini_mixture_of_5_diverse` example!
For example, you could try mixing candidates that use different LLMs.

## Next Steps

You now have a ClickHouse database with a ton of trajectories of LLMs trying to solve chess puzzles.
Consider our library of [recipes](https://www.tensorzero.com/docs/recipes) for ideas on how to use this dataset to improve further!
Since this data ended up in ClickHouse, we also included a test set at `data/lichess_easy_puzzles_test.csv` (use `dryrun=True` to avoid leaking it) to evaluate variants on held-out data.
