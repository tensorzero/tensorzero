import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import chess
import chess.engine
import chess.pgn
import pandas as pd
from scipy.stats import binomtest
from tensorzero import AsyncTensorZeroGateway
from tqdm import trange

log = logging.getLogger(__name__)


class AbstractPlayer(ABC):
    @abstractmethod
    async def play(
        self, board: chess.Board, episode_id: Optional[UUID] = None
    ) -> Tuple[str, Optional[UUID]]:
        pass


class StockfishPlayer(AbstractPlayer):
    """
    If you have [Stockfish](https://stockfishchess.org/) installed and want to see how a real chess engine performs, use this player instead.
    """

    def __init__(self, player_elo: int):
        assert player_elo >= 1320 and player_elo <= 3190
        self.player_elo = player_elo
        self.player_engine = None

    async def _initialize(self):
        _transport_player, player_engine = await chess.engine.popen_uci("stockfish")
        await player_engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": self.player_elo}
        )
        self.player_engine = player_engine

    async def play(
        self, board: chess.Board, episode_id: Optional[UUID] = None
    ) -> Tuple[str, Optional[UUID]]:
        if self.player_engine is None:
            await self._initialize()
        result = await self.player_engine.play(board, chess.engine.Limit(time=0.2))
        return board.san(result.move), None


class TensorZeroPlayer(AbstractPlayer):
    def __init__(
        self, client: AsyncTensorZeroGateway, variant_name: Optional[str] = None
    ):
        self.client = client
        self.variant_name = variant_name

    async def play(
        self, board: chess.Board, episode_id: Optional[UUID] = None
    ) -> Tuple[str, Optional[UUID]]:
        legal_moves_san = [board.san(move) for move in board.legal_moves]

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
                episode_id=episode_id,
            )
            thinking = result.output.parsed["thinking"]
            log.info(f"Player thinking: {thinking}")
            move = result.output.parsed["move"]
            log.info(f"Player move: {move}")
            episode_id = result.episode_id
        except Exception as e:
            log.error(f"Error occurred: {e}")
            log.info("Choosing a random legal move as fallback.")
            move = random.choice(legal_moves_san)
            return move, episode_id
        return move, episode_id


async def run_puzzle(
    puzzle_data: Dict, player: AbstractPlayer, semaphore: asyncio.Semaphore
) -> Tuple[bool, Optional[UUID]]:
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
    log.info(f"Initial move applied: {first_move_obj}\n")
    move_index = 1

    # Determine player's color based on the updated position
    player_color = board.turn  # True for White, False for Black
    episode_id = None
    while move_index < total_moves and not board.is_game_over():
        if board.turn == player_color:
            # Player's move
            log.info(f"Puzzle ID: {puzzle_id}")
            log.info(f"Current Board:\n{board}\n")
            log.info(f"Player color: {'White' if player_color else 'Black'}")
            async with semaphore:
                player_move_san, episode_id = await player.play(
                    deepcopy(board), episode_id
                )
            expected_move = expected_moves[move_index]

            try:
                player_move_obj = board.parse_san(player_move_san)
            except ValueError:
                log.info(f"Invalid SAN move format: {player_move_san}")
                return False, episode_id

            try:
                expected_move_obj = board.parse_san(expected_move)
            except ValueError:
                expected_move_obj = chess.Move.from_uci(expected_move)

            if board.is_checkmate():
                log.info("Player has delivered a checkmate!")
                return True, episode_id

            if player_move_obj != expected_move_obj:
                log.info(
                    f"Incorrect move at move {move_index + 1}: expected {expected_move}, got {player_move_san}"
                )
                return False, episode_id

            board.push(player_move_obj)
            log.info(f"Player moved: {player_move_obj}\n")
        else:
            # Opponent's move
            expected_move = expected_moves[move_index]
            try:
                opponent_move_obj = board.parse_san(expected_move)
            except ValueError:
                opponent_move_obj = chess.Move.from_uci(expected_move)

            board.push(opponent_move_obj)
            log.info(f"Opponent moved: {opponent_move_obj}\n")

        move_index += 1

    if move_index == total_moves:
        log.info("Player successfully completed all expected moves in the puzzle!")
        return True, episode_id
    else:
        log.info("Player failed to complete the puzzle as expected.")
        return False, episode_id


def proportion_ci(
    successes: int, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    low, high = binomtest(successes, trials).proportion_ci(confidence_level=confidence)
    return low, high
