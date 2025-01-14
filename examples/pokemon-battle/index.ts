const { BattleStream, BattlePlayer, getPlayerStreams } = require('pokemon-showdown/dist/sim/battle-stream.js');
const { RandomPlayerAI } = require('pokemon-showdown/dist/sim/tools/random-player-ai.js');
const { Teams } = require('pokemon-showdown/dist/sim/teams.js');
const path = require('path');
const fs = require('fs');

// Note - type-checking for 'pokemon-showdown' doesn't work correctly (it's missing '.d.ts' files),
// so we just define some minimal types here, prefixed with 'I' to avoid conflicts with 'pokemon-showdown'

interface IBattleStream {
    battle: IBattle;
}

interface IBattle {
    getDebugLog(): string;
}

interface IBattleRequest {
    wait: boolean;
    teamPreview: boolean;
    side: ISide;
}

interface IPokemon {
    ident: string;
    condition: string;
}

interface ISide {
    pokemon: IPokemon[];
}

const battleStream: IBattleStream = new BattleStream();
const streams = getPlayerStreams(battleStream);

const TENSOR_FUNCTION = "poke_battle";
const TENSOR_METRIC = "poke_battle_win";
const TENSOR_BOT_NAME = "TensorZeroBot";
const RANDOM_BOT_NAME = "RandomBot";


// Used fixed teams to allow for easy fine-tuning
const teamZero = [{ "name": "Nidoking", "species": "Nidoking", "moves": ["blizzard", "rockslide", "earthquake", "substitute"], "ability": "No Ability", "evs": { "hp": 251, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }, "ivs": { "hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30 }, "item": "", "level": 74, "shiny": false, "gender": false }, { "name": "Doduo", "species": "Doduo", "moves": ["bodyslam", "drillpeck", "agility", "doubleedge"], "ability": "No Ability", "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }, "ivs": { "hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30 }, "item": "", "level": 87, "shiny": false, "gender": false }]
const teamOne = [{ "name": "Snorlax", "species": "Snorlax", "moves": ["amnesia", "bodyslam", "blizzard", "selfdestruct"], "ability": "No Ability", "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }, "ivs": { "hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30 }, "item": "", "level": 69, "shiny": false, "gender": false }, { "name": "Golduck", "species": "Golduck", "moves": ["blizzard", "amnesia", "surf", "hydropump"], "ability": "No Ability", "evs": { "hp": 255, "atk": 0, "def": 255, "spa": 255, "spd": 255, "spe": 255 }, "ivs": { "hp": 30, "atk": 2, "def": 30, "spa": 30, "spd": 30, "spe": 30 }, "item": "", "level": 75, "shiny": false, "gender": false }]

const teams = [teamZero, teamOne];
const teamId = (Math.random() < 0.5) ? 0 : 1;

// Randomly assign the teams to the players
const randomBotTeam = JSON.parse(JSON.stringify(teams[teamId]));
const tensorZeroTeam = JSON.parse(JSON.stringify(teams[1 - teamId]));

// We nerf the HP the pokemon in the tensorZero team, to make things harder for the model.
for (var pokemon of tensorZeroTeam) {
    pokemon["evs"]["hp"] = 0;
    pokemon["ivs"]["hp"] = 0;
}

// Copied from https://github.com/smogon/pokemon-showdown/blob/920c6f3e5c54c21c5642cc65fcb76fe8ecef72d2/test/common.js#L131
// which is not exported from the package
function saveReplay(battle: IBattle, fileName: string) {
    const battleLog = battle.getDebugLog();
    if (!fileName) fileName = 'test-replay';
    const filePath = path.resolve(`./replays/${fileName}-${Date.now()}.html`);
    const out = fs.createWriteStream(filePath, { flags: 'a' });
    out.on('open', () => {
        out.write(
            `<!DOCTYPE html>\n` +
            `<script type="text/plain" class="battle-log-data">${battleLog}</script>\n` +
            `<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>\n`
        );
        out.end();
    });
    return filePath;
}

class TensorZeroAI extends BattlePlayer {
    pokemonState: Record<string, string> = {};
    winner: string | null = null;
    lastRequest: any = null;

    constructor(
        playerStream: any,
        debug = false
    ) {
        super(playerStream, debug);
    }

    async tensorZeroRequest(body: any, error: string | null) {
        const full_req = {
            "request": body,
            "context": this.pokemonState,
            "prevError": error
        };
        console.log("Making request: ", JSON.stringify(full_req));
        const resp = await fetch('http://localhost:3000/inference', {
            method: 'POST',
            body: JSON.stringify({
                "function_name": TENSOR_FUNCTION,
                "episode_id": this.episodeId,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": full_req,
                        }
                    ]
                }
            })
        });
        const json = await resp.json();
        console.log("Got response: ", json);
        if (this.episodeId == null) {
            this.episodeId = json["episode_id"];
            console.log("Set episode id to: " + this.episodeId);
        } else if (this.episodeId !== json["episode_id"]) {
            throw new Error("Episode id changed from " + this.episodeId + " to " + json["episode_id"]);
        }
        return json["output"]["parsed"]
    }

    async recordWin(tensorZeroWin: boolean) {
        const resp = await fetch('http://localhost:3000/feedback', {
            method: 'POST',
            body: JSON.stringify({
                "metric_name": TENSOR_METRIC,
                "episode_id": this.episodeId,
                "value": !!tensorZeroWin,
                "tags": {
                    "team_id": `${teamId}`
                }
            })
        });
        if (resp.ok) {
            console.log(`Recorded win=${tensorZeroWin} with team_id ${teamId}`);
        } else {
            console.error("Failed to record metric: ", resp);
        }
    }

    updateStateFromLine(lineIdent: string, status: string) {
        // Normalize the identifier to match the information that we get from 'receiveRequest'
        lineIdent = lineIdent.replace("p1a:", "p1:").replace("p2a:", "p2:");
        this.pokemonState[lineIdent] = status;
    }

    receiveLine(line: string) {
        super.receiveLine(line);
        if (line.startsWith('|-damage') || line.startsWith('|-heal') || line.startsWith('-sethp')) {
            const [_a, _b, pokemon, health] = line.split('|');
            this.updateStateFromLine(pokemon, health);
        }
        if (line.startsWith('|faint')) {
            const [_a, _b, pokemon] = line.split('|');
            this.updateStateFromLine(pokemon, "0 fnt");
        }
        if (line.startsWith('|switch')) {
            const [_a, _b, pokemon, _c, health] = line.split('|');
            console.log("Switching to: ", pokemon, health);
            this.updateStateFromLine(pokemon, health);
        }
        if (line.startsWith('|win')) {
            this.winner = line.split('|')[2];
        }
    }

    receiveError(error: Error) {
        var errMsg = error.toString();
        console.info(`Got error error: '${errMsg}'`);
        // When we get an error, let the model try again, giving it the error messages
        /// so that it can attempt to pick a different, valid choice.
        this.tensorZeroRequest(this.lastRequest, errMsg).then(async resp => {
            this.choose(resp.action);
        })
    }

    receiveRequest(request: IBattleRequest) {
        for (const pokemon of request.side.pokemon) {
            this.pokemonState[pokemon.ident] = pokemon.condition;
        }
        if (request.wait) {
            // wait request
            // do nothing
            return;
        }

        if (request.teamPreview) {
            return this.choose("default");
        }

        this.tensorZeroRequest(request, null).then(async resp => {
            this.lastRequest = JSON.parse(JSON.stringify(request));
            this.choose(resp.action);
        })
    }

    async start() {
        await super.start();
        const ourPokemon = [];
        const enemyPokemon = [];
        for (const [ident, condition] of Object.entries(this.pokemonState)) {
            if (ident.startsWith("p1:")) {
                ourPokemon.push([ident, condition]);
            } else {
                enemyPokemon.push([ident, condition]);
            }
        }
        ourPokemon.sort();
        enemyPokemon.sort();
        console.log("Winner: ", this.winner + " with team id " + teamId);
        console.log("Our pokemon: ", ourPokemon);
        console.log("Enemy pokemon: ", enemyPokemon);

        const win = this.winner === TENSOR_BOT_NAME;
        const winLoss = win ? "win" : "loss";

        const replayFile = saveReplay(battleStream.battle, `tensorBattle-${winLoss}-teamId-${teamId}`);
        console.log("Saved replay to: ", replayFile);
        this.recordWin(win);
    }
}

const spec = {
    formatid: "gen7customgame",
    strictChoices: false
};

const randomTeam = Teams.generate('gen1randombattle').slice(0, 2);


const tensorBotSpec = {
    name: TENSOR_BOT_NAME,
    team: Teams.pack(tensorZeroTeam),
};
const randomBotSpec = {
    name: RANDOM_BOT_NAME,
    team: Teams.pack(randomBotTeam),
};

const p1 = new TensorZeroAI(streams.p1);
const p2 = new RandomPlayerAI(streams.p2);

console.log("p1 is " + p1.constructor.name);
console.log("p2 is " + p2.constructor.name);

void p1.start();
void p2.start();

void (async () => {
    for await (const chunk of streams.omniscient) {
        console.log(chunk);
    }
})();

// Note - the 'p1' and 'p2' assignments must match the 'streams.p1' and 'streams.p2' usages above
void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(tensorBotSpec)}
>player p2 ${JSON.stringify(randomBotSpec)}`);
