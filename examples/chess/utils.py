import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import pandas as pd
from chess.engine import UciProtocol
from tensorzero import AsyncTensorZeroGateway
from tqdm import trange


class AbstractPlayer(ABC):
    @abstractmethod
    async def play(self, board: chess.Board) -> str:
        pass


class StockfishPlayer(AbstractPlayer):
    def __init__(self, player_engine: UciProtocol):
        self.player_engine = player_engine

    async def play(self, board: chess.Board) -> str:
        legal_moves_san = [board.san(move) for move in board.legal_moves]
        print(legal_moves_san)
        print(board)
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
        # print(f"pgn: {pgn}")

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
        self.episode_id = result.episode_id
        thinking = result.output.parsed["thinking"]
        print(f"Player thinking: {thinking}")
        move = result.output.parsed["move"]
        print(f"Player move: {move}")
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
        print(board)
        print()
        # Evaluate the position
        info = await game_engine.analyse(board, chess.engine.Limit(time=0.1))
        print(f"Evaluation: {info['score']}")
        print()
        move = await player.play(deepcopy(board))
        board.push_san(move)

        if board.is_game_over():
            break

        print(board)
        print()
        # Evaluate the position
        info = await game_engine.analyse(board, chess.engine.Limit(time=0.1))
        print(f"Evaluation: {info['score']}")
        print()
        # Game engine's move (Black)
        result = await game_engine.play(deepcopy(board), chess.engine.Limit(time=0.2))
        board.push(result.move)
    print(board)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        print("Game over! Draw.")
    else:
        winner_name = "White" if outcome.winner else "Black"
        print(f"Game over! {winner_name} wins.")

    # Quit the game engine
    await game_engine.quit()


class PuzzleChecker:
    def __init__(self, puzzle_data: dict):
        self.puzzle_id = puzzle_data["PuzzleId"]
        self.fen = puzzle_data["FEN"]
        self.expected_moves = puzzle_data["Moves"].split()
        self.rating = puzzle_data["Rating"]
        self.rating_deviation = puzzle_data["RatingDeviation"]
        self.popularity = puzzle_data["Popularity"]
        self.nb_plays = puzzle_data["NbPlays"]
        self.themes = puzzle_data["Themes"].split()
        self.game_url = puzzle_data["GameUrl"]

    async def run_puzzle(self, player: AbstractPlayer) -> bool:
        board = chess.Board(self.fen)
        move_index = 0
        total_moves = len(self.expected_moves)

        # Apply the first move before starting the puzzle
        first_move = self.expected_moves[move_index]
        try:
            first_move_obj = board.parse_san(first_move)
        except ValueError:
            first_move_obj = chess.Move.from_uci(first_move)
        board.push(first_move_obj)
        print(f"Initial move applied: {first_move_obj}\n")
        move_index = 1

        # Determine player's color based on the updated position
        player_color = board.turn  # True for White, False for Black

        while move_index < total_moves and not board.is_game_over():
            if board.turn == player_color:
                # Player's move
                print(f"Puzzle ID: {self.puzzle_id}")
                print(f"Current Board:\n{board}\n")
                player_move_san = await player.play(deepcopy(board))
                expected_move = self.expected_moves[move_index]

                try:
                    player_move_obj = board.parse_san(player_move_san)
                except ValueError:
                    print(f"Invalid SAN move format: {player_move_san}")
                    return False

                try:
                    expected_move_obj = board.parse_san(expected_move)
                except ValueError:
                    expected_move_obj = chess.Move.from_uci(expected_move)
                if board.is_checkmate():
                    print("Player has delivered a checkmate!")
                    return True

                if player_move_obj != expected_move_obj:
                    print(
                        f"Incorrect move at move {move_index + 1}: expected {expected_move}, got {player_move_san}"
                    )
                    return False

                board.push(player_move_obj)
                print(f"Player moved: {player_move_obj}\n")
            else:
                # Opponent's move
                expected_move = self.expected_moves[move_index]
                try:
                    opponent_move_obj = board.parse_san(expected_move)
                except ValueError:
                    opponent_move_obj = chess.Move.from_uci(expected_move)

                board.push(opponent_move_obj)
                print(f"Opponent moved: {opponent_move_obj}\n")

            move_index += 1

        if move_index == total_moves:
            print("Player successfully completed all expected moves in the puzzle!")
            return True
        else:
            print("Player failed to complete the puzzle as expected.")
            return False


def update_puzzle_elo(
    current_elo: float, puzzle_elo: float, puzzle_number: int, success: bool
) -> float:
    expected_score = 1 / (1 + 10 ** ((puzzle_elo - current_elo) / 400))
    k_factor = 40 if puzzle_number < 30 else 20
    actual_score = 1 if success else 0
    return current_elo + k_factor * (actual_score - expected_score)


async def estimate_player_puzzle_elo(
    player: AbstractPlayer, puzzle_df: pd.DataFrame, num_puzzles: int = 100
) -> Tuple[float, List[float]]:
    rating = 1500
    rating_history = []
    for i in trange(num_puzzles):
        # Get a random puzzle within 20 points of the current rating
        rating_range = (rating - 20, rating + 20)
        eligible_puzzles = puzzle_df[
            (puzzle_df["Rating"] >= rating_range[0])
            & (puzzle_df["Rating"] <= rating_range[1])
        ]

        if eligible_puzzles.empty:
            # If no puzzles in range, get the closest one
            closest_puzzle = puzzle_df.iloc[
                (puzzle_df["Rating"] - rating).abs().argsort()[:1]
            ]
        else:
            closest_puzzle = eligible_puzzles.sample(n=1)

        random_puzzle = closest_puzzle.iloc[0].to_dict()

        puzzle_checker = PuzzleChecker(random_puzzle)
        print(f"Running puzzle {i+1} with rating {random_puzzle['Rating']}")
        success = await puzzle_checker.run_puzzle(player)

        rating = update_puzzle_elo(rating, random_puzzle["Rating"], i + 1, success)

        print(
            f"Puzzle {i+1} completed {'successfully' if success else 'unsuccessfully'}. New estimated Elo: {rating:.2f}"
        )
        rating_history.append(rating)
    return rating, rating_history


async def main(game_elo: int = 1320) -> None:
    # Initialize player engine
    # transport_player, player_engine = await chess.engine.popen_uci("stockfish")
    # player_elo = 1320
    # await player_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": player_elo})
    # player = StockfishPlayer(player_engine=player_engine)
    puzzle_df = pd.read_csv("data/lichess_popular_puzzles.csv")
    async with AsyncTensorZeroGateway("http://localhost:3000") as client:
        player = TensorZeroPlayer(client=client, variant_name="gpt-4o-mini_best_of_5")
        estimated_elo, rating_history = await estimate_player_puzzle_elo(
            player, puzzle_df
        )
    print(f"Estimated Elo: {estimated_elo:.2f}")
    plt.plot(rating_history)
    plt.xlabel("Puzzle Number")
    plt.ylabel("Estimated Elo")
    plt.title("Estimated Elo History")
    plt.show()
    # Quit the player engine
    # await player_engine.quit()


if __name__ == "__main__":
    asyncio.run(main())
