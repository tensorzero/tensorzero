import asyncio
import logging
import random
from abc import ABC, abstractmethod
import json
from copy import deepcopy
from typing import Dict, List, Optional

import chess
import chess.engine
import chess.pgn
import pandas as pd
from chess.engine import UciProtocol
from tensorzero import AsyncTensorZeroGateway
from tqdm import trange

logger = logging.getLogger(__name__)


class AbstractPlayer(ABC):
    @abstractmethod
    async def play(self, board: chess.Board) -> str:
        pass


class StockfishPlayer(AbstractPlayer):
    def __init__(self, player_engine: UciProtocol):
        self.player_engine = player_engine

    async def play(self, board: chess.Board) -> str:
        legal_moves_san = [board.san(move) for move in board.legal_moves]
        logger.info(legal_moves_san)
        logger.info(board)
        result = await self.player_engine.play(board, chess.engine.Limit(time=0.2))
        return board.san(result.move)


class TensorZeroPlayer(AbstractPlayer):
    def __init__(
        self, client: AsyncTensorZeroGateway, variant_name: Optional[str] = None
    ):
        self.client = client
        self.variant_name = variant_name
        self.episode_id = None

    async def play(self, board: chess.Board) -> str:
        legal_moves_san = [board.san(move) for move in board.legal_moves]
        # pgn = _get_pgn_from_board(board)
        # logger.info(f"pgn: {pgn}")

        try:
            result = await self.client.inference(
                function_name="play_chess_board",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "board": str(board),
                                "color": "white" if board.turn else "black",
                                "legal_moves_san": legal_moves_san,
                            },
                        }
                    ]
                },
                variant_name=self.variant_name,
                episode_id=self.episode_id,
            )
            thinking = result.output.parsed["thinking"]
            logger.info(f"Player thinking: {thinking}")
            move = result.output.parsed["move"]
            logger.info(f"Player move: {move}")
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            logger.info("Choosing a random legal move as fallback.")
            move = random.choice(legal_moves_san)
            return move
        self.episode_id = result.episode_id
        return move


def _get_pgn_from_board(board: chess.Board) -> str:
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter)
    return pgn_string


async def play_game(
    player: AbstractPlayer,
    game_elo: int,
) -> None:
    # Initialize game engine
    transport_game, game_engine = await chess.engine.popen_uci("stockfish")
    await game_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": game_elo})

    board = chess.Board()
    while not board.is_game_over():
        # Player's move (White)
        logger.info(board)
        logger.info("")
        # Evaluate the position
        info = await game_engine.analyse(board, chess.engine.Limit(time=0.1))
        logger.info(f"Evaluation: {info['score']}")
        logger.info("")
        move = await player.play(deepcopy(board))
        board.push_san(move)

        if board.is_game_over():
            break

        logger.info(board)
        logger.info("")
        # Evaluate the position
        info = await game_engine.analyse(board, chess.engine.Limit(time=0.1))
        logger.info(f"Evaluation: {info['score']}")
        logger.info("")
        # Game engine's move (Black)
        result = await game_engine.play(deepcopy(board), chess.engine.Limit(time=0.2))
        board.push(result.move)
    logger.info(board)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        logger.info("Game over! Draw.")
    else:
        winner_name = "White" if outcome.winner else "Black"
        logger.info(f"Game over! {winner_name} wins.")

    # Quit the game engine
    await game_engine.quit()


async def run_puzzle(
    puzzle_data: Dict, player: AbstractPlayer, semaphore: asyncio.Semaphore
) -> bool:
    """
    Runs a chess puzzle for the given player and checks if the player solves it correctly.

    Args:
        puzzle_data (Dict): A dictionary containing puzzle details.
        player (AbstractPlayer): An instance of a player that can make moves.

    Returns:
        bool: True if the player solves the puzzle correctly, False otherwise.
    """
    # Extract puzzle details from puzzle_data
    puzzle_id = puzzle_data.get("PuzzleId")
    fen = puzzle_data.get("FEN")
    expected_moves = puzzle_data.get("Moves", "").split()

    board = chess.Board(fen)
    move_index = 0
    total_moves = len(expected_moves)

    # Apply the first move before starting the puzzle
    first_move = expected_moves[move_index]
    first_move_obj = board.parse_san(first_move)
    board.push(first_move_obj)
    logger.info(f"Initial move applied: {first_move_obj}\n")
    move_index = 1

    # Determine player's color based on the updated position
    player_color = board.turn  # True for White, False for Black

    while move_index < total_moves and not board.is_game_over():
        if board.turn == player_color:
            # Player's move
            logger.info(f"Puzzle ID: {puzzle_id}")
            logger.info(f"Current Board:\n{board}\n")
            logger.info(f"Player color: {'White' if player_color else 'Black'}")
            async with semaphore:
                player_move_san = await player.play(deepcopy(board))
            expected_move = expected_moves[move_index]

            try:
                player_move_obj = board.parse_san(player_move_san)
            except ValueError:
                logger.info(f"Invalid SAN move format: {player_move_san}")
                return False

            try:
                expected_move_obj = board.parse_san(expected_move)
            except ValueError:
                expected_move_obj = chess.Move.from_uci(expected_move)

            if board.is_checkmate():
                logger.info("Player has delivered a checkmate!")
                return True

            if player_move_obj != expected_move_obj:
                logger.info(
                    f"Incorrect move at move {move_index + 1}: expected {expected_move}, got {player_move_san}"
                )
                return False

            board.push(player_move_obj)
            logger.info(f"Player moved: {player_move_obj}\n")
        else:
            # Opponent's move
            expected_move = expected_moves[move_index]
            try:
                opponent_move_obj = board.parse_san(expected_move)
            except ValueError:
                opponent_move_obj = chess.Move.from_uci(expected_move)

            board.push(opponent_move_obj)
            logger.info(f"Opponent moved: {opponent_move_obj}\n")

        move_index += 1

    if move_index == total_moves:
        logger.info("Player successfully completed all expected moves in the puzzle!")
        return True
    else:
        logger.info("Player failed to complete the puzzle as expected.")
        return False


async def estimate_player_puzzle_elo(
    player: AbstractPlayer,
    puzzle_df: pd.DataFrame,
    variant_name: str,
    semaphore: asyncio.Semaphore,
) -> List[bool]:
    successes = []
    num_successes = 0
    total_puzzles = len(puzzle_df)
    progress_bar = trange(total_puzzles, desc=variant_name)

    tasks = [
        asyncio.create_task(run_puzzle(puzzle_df.iloc[i].to_dict(), player, semaphore))
        for i in range(total_puzzles)
    ]

    for task in asyncio.as_completed(tasks):
        success = await task
        successes.append(success)
        if success:
            num_successes += 1
        current = len(successes)
        logger.info(
            f"Puzzle {current} completed {'successfully' if success else 'unsuccessfully'}"
        )
        p = num_successes / current
        z = 1.96  # for 95% confidence
        se = (p * (1 - p) / current) ** 0.5
        ci_lower = max(0, p - z * se)
        ci_upper = min(1, p + z * se)
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "Success": f"{num_successes}/{current} CI: ({ci_lower:.2f}, {ci_upper:.2f})"
            },
            refresh=True,
        )

    progress_bar.close()
    return successes


async def get_variant_rating_history(
    client: AsyncTensorZeroGateway,
    puzzle_df: pd.DataFrame,
    variant_name: str,
) -> List[bool]:
    player = TensorZeroPlayer(client=client, variant_name=variant_name)
    max_concurrent_requests = 10
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    successes = await estimate_player_puzzle_elo(
        player, puzzle_df, variant_name, semaphore
    )
    return successes


async def main() -> None:
    # Initialize player engine
    # transport_player, player_engine = await chess.engine.popen_uci("stockfish")
    # player_elo = 1320
    # await player_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": player_elo})
    # player = StockfishPlayer(player_engine=player_engine)
    variants_to_evaluate = [
        "gpt-4o-mini",
        "gpt-4o-mini_best_of_5",
    ]
    puzzle_df = pd.read_csv("data/lichess_easy_puzzles_test.csv")
    puzzle_df = puzzle_df.head(1000)
    async with AsyncTensorZeroGateway("http://localhost:3000", timeout=30.0) as client:
        tasks = [
            get_variant_rating_history(client, puzzle_df, variant)
            for variant in variants_to_evaluate
        ]
        successes = await asyncio.gather(*tasks)
        print("\n\n\n\n")

        for variant, success in zip(variants_to_evaluate, successes):
            num_successes = sum(success)
            total_puzzles = len(success)
            if total_puzzles > 0:
                print(
                    f"{variant}: {num_successes}/{total_puzzles} = {num_successes/total_puzzles:.2f}"
                )
                p = num_successes / total_puzzles
                z = 1.96  # for 95% confidence
                se = (p * (1 - p) / total_puzzles) ** 0.5
                ci_lower = max(0, p - z * se)
                ci_upper = min(1, p + z * se)
                print(
                    f"{variant} confidence interval: ({ci_lower:.2f}, {ci_upper:.2f})"
                )
    # Dump JSON of variant name to successes and total
    variant_results = {
        variant: {"successes": sum(success), "total": len(success)}
        for variant, success in zip(variants_to_evaluate, successes)
    }

    with open("variant_results.json", "w") as f:
        json.dump(variant_results, f, indent=4)

    print("Results have been saved to variant_results.json")
    # Quit the player engine
    # await player_engine.quit()


if __name__ == "__main__":
    asyncio.run(main())
