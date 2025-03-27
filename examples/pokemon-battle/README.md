# Example: Creating a Pokemon Showdown battle bot with TensorZero

![battle](./battle.png)

## Background

The [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) project can be used to simulate Pokemon battles.
In this project, we implement an LLM-based bot, which chooses in-game moves based on the current state of the battle.
Using TensorZero, we record all inferences made during a battle as a single episode, and record a "poke_battle_win" metric to indicate whether or not the bot won the battle.

Each battle creates an HTML file in the 'replays' directory, which can be viewed in your browser to see a replay of the battle.

Our bot plays against a simple "random bot", which just makes random moves.

To reduce the variance of the results, we use two fixed teams of Pokemon, each containing only two pokemon.
Each team is randomly assigned to either the TensorZero bot or the random bot. Additionally, we "nerf" the stats of the pokemon on whichever team is assigned to TensorZero, to make things more interesting.


### Future Work
Over time, we'll be expanding the scope of this bot
* Support for battling online on the Pokemon Showdown server
* Configuration options for the team size and pokemon generation/stats
* Support for battling against other types of bots, not just a random bot
* More details metrics (e.g. health remaining, number of fainted pokemon, etc.)


## Setup

### TensorZero

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and run the following command.
Docker Compose will launch the TensorZero gateway and a test ClickHouse database.

```bash
docker compose up
```

You can run a single battle with `npm run-script run`


On Linux-based systems (with gnu `parallel` installed), you can use `./multi.sh` to spawn 100 instances of the bot in parallel.

### Viewing the Results

You can view the results in the ClickHouse database.
The `./win_stats.sh` script will print out a mean and confidence interval for the win rate of the OpenAI-based bot against the random bot:

Here's the results sample run:

```
   ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ team_id ┃          mean_value ┃ sample_size ┃           ci_lower ┃           ci_upper ┃
   ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
1. │ 1       │  0.6086956521739131 │          46 │ 0.4676581551269506 │ 0.7497331492208756 │
   ├─────────┼─────────────────────┼─────────────┼────────────────────┼────────────────────┤
2. │ 0       │ 0.48148148148148145 │          54 │ 0.3482118731767807 │ 0.6147510897861822 │
   └─────────┴─────────────────────┴─────────────┴────────────────────┴────────────────────┘
```

When playing with team 0 (Nidoking and Doduo), the OpenAI-based bot wins ~60% of the time.
When playing with team 1 (Snorlax and Golduck), the OpenAI-based bot wins ~48% of the time.