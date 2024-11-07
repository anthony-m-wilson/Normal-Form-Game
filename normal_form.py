"""
Normal Form Game

This program is designed to provide differen solutions for a normal form game. 
The solution include different strategies such as: 
weakly and strongly dominated, nash equilibrium, pareto optimal outcomes, minimax, and maximin.
It also allows you to test the strategies in a repetitive game to see the average scoring for each player.

To run program use the following command: python normal_form.py
The code will run through all the files and provide the solutions for each game. 
The code assumes that the starting files are the same as the ones provided in the assignment zip: prog3A.txt, prog3B.txt, prog3C.txt, prog3D.txt
If a simulation is desired for the repetitive game, set the repeated variable to True for start_game function in the main call.

Author: Anthony Wilson
Date: 2/22/2024
"""

import random
import sys


class NormalForm:
    def __init__(self, file):
        self.file = file
        self.row_length = 0
        self.column_length = 0
        self.row_values = []
        self.column_values = []
        self.game_matrix = []

    def create_game_matrix(self, file):
        f = open(file, "r")
        init = f.readline()

        self.row_length, self.column_length = init.strip().split(" ")
        self.row_length = int(self.row_length)
        self.column_length = int(self.column_length)

        self.row_values = f.readline().strip().split(" ")
        self.column_values = f.readline().strip().split(" ")

        row_length = len(self.row_values) // self.row_length

        # Create game matrix of tuples (row, column)
        self.game_matrix = list(
            tuple(
                zip(list(map(int, self.row_values)), list(map(int, self.column_values)))
            )
        )

        # split seperate rows
        self.game_matrix = [
            self.game_matrix[i : i + row_length]
            for i in range(0, len(self.game_matrix), row_length)
        ]

        self.print_game_matrix(file)

    def print_game_matrix(self, file):
        length = (self.column_length + 1) * 11
        print()
        print(f'-Game Matrix for file "{file}"-'.center(length, " "))

        print("".ljust(length + 1, "-"))
        print("|", end="")
        print("P1 \\ P2".center(10, " "), end="|")

        for col in range(self.column_length):
            print(f"{col}".center(10, " "), end="|")
        print("\n".ljust(length + 2, "-"))

        for row in range(self.row_length):
            print("|", end="")
            print(f"{row}".center(10, " "), end="|")
            for col in range(self.column_length):
                print(f"{self.game_matrix[row][col]}".center(10, " "), end="|")
            print("\n".ljust(length + 2, "-"))

    def solve_dominated_strategies(self, start_player, weak):
        # Start with all rows and columns
        remaining_rows = set(range(len(self.game_matrix)))
        remaining_cols = set(range(len(self.game_matrix[0])))

        while True:
            # Eliminate dominated strategies
            # If a eliminated strategy was found, repeat the process to find more
            eliminated = self.eliminate_dominated_strategies(
                start_player, weak, remaining_rows, remaining_cols
            )
            # If no strategy was eliminated in this iteration, we are done
            if not eliminated:
                print("Strategy:", "Weak" if weak else "Strong")
                print("Start Player:", start_player)
                print("  Remaining Rows:", remaining_rows)
                print("  Remaining Cols:", remaining_cols)
                break

        # Return remaining strategies as indices
        if len(remaining_rows) == 1 and len(remaining_cols) == 1:
            return [(i, j) for i in remaining_rows for j in remaining_cols][0]
        else:
            return None

    def eliminate_dominated_strategies(self, player, weak, rows, cols):
        # No values have been eliminated yet
        eliminated = False

        # Eliminate weakly dominated strategies
        if weak:
            if player == 1:
                # If starting with player 1, eliminate row first, then column
                eliminated = self.find_dominated_strategies(1, True, rows, cols)
                eliminated = self.find_dominated_strategies(2, True, rows, cols)
            else:
                # If starting with player 2, eliminate column first, then row
                eliminated = self.find_dominated_strategies(2, True, rows, cols)
                eliminated = self.find_dominated_strategies(1, True, rows, cols)

        # Eliminate strongly dominated strategies
        else:
            if player == 1:
                # If starting with player 1, eliminate row first, then column
                eliminated = self.find_dominated_strategies(1, False, rows, cols)
                eliminated = self.find_dominated_strategies(2, False, rows, cols)
            else:
                # If starting with player 2, eliminate column first, then row
                eliminated = self.find_dominated_strategies(2, False, rows, cols)
                eliminated = self.find_dominated_strategies(1, False, rows, cols)

        # If any values were eliminated, return True
        return eliminated

    def find_dominated_strategies(self, player, weak, rows, cols):
        # Start with an empty set of eliminated values
        eliminated_values = set()
        eliminated = False

        # If player 1, compare row
        if player == 1:
            for i in list(rows):
                for j in list(rows):
                    if i != j:
                        dominates = (
                            # (weak) Find values that are equal or greater than the others
                            all(
                                self.game_matrix[j][k][0] >= self.game_matrix[i][k][0]
                                for k in cols
                            )
                            if weak
                            else all(
                                # (Strong) Find values that are greater than the others
                                self.game_matrix[j][k][0] > self.game_matrix[i][k][0]
                                for k in cols
                            )
                        )
                        # If there was a match, eliminate the value from rows
                        if dominates:
                            eliminated_values.add(i)
                            rows.remove(i)
                            eliminated = True
                            break
            # Return true or false if any values were eliminated
            return eliminated

        # If player 2, compare column
        else:
            for k in list(cols):
                for l in list(cols):
                    if k != l:
                        dominates = (
                            all(
                                # (weak) Find values that are equal or greater than the others
                                self.game_matrix[i][l][1] >= self.game_matrix[i][k][1]
                                for i in rows
                            )
                            if weak
                            else all(
                                # (Strong) Find values that are greater than the others
                                self.game_matrix[i][l][1] > self.game_matrix[i][k][1]
                                for i in rows
                            )
                        )
                        # If there was a match, eliminate the value from columns
                        if dominates:
                            eliminated_values.add(k)
                            cols.remove(k)
                            eliminated = True
                            break
            # Return true or false if any values were eliminated
            return eliminated

    def find_nash_equilibrium(self):
        nash_equilibria = []

        # for each cell
        for i in range(len(self.game_matrix)):
            for j in range(len(self.game_matrix[i])):
                payoff_pair = self.game_matrix[i][j]
                # Find player 1's values for each column
                column_values = [
                    self.game_matrix[k][j][0] for k in range(len(self.game_matrix))
                ]
                # Find player 2's values for each row
                row_values = [
                    self.game_matrix[i][l][1] for l in range(len(self.game_matrix[i]))
                ]

                # If the current cell max in both row and column, it is a nash equilibrium
                if payoff_pair[0] == max(column_values) and payoff_pair[1] == max(
                    row_values
                ):
                    # Possiblitity for multiple nash equilibria, so add to a list
                    nash_equilibria.append((i, j))

        if nash_equilibria:
            for eq in nash_equilibria:
                print(
                    f"Nash Equilibrium at index {eq} with value {self.game_matrix[eq[0]][eq[1]]}"
                )
        else:
            print("No Nash Equilibrium found")

    def pareto_optimal_outcomes(self):
        pareto_optimal_outcomes = []

        # for each cell
        for i in range(len(self.game_matrix)):
            for j in range(len(self.game_matrix[i])):
                payoff_pair = self.game_matrix[i][j]

                # Checks for cell happiness of both players when the other is fixed
                # To be pareto optimal, the cell must not have any better options for either player
                if not any(
                    all(
                        payoff_pair[k] <= other_payoff_pair[k]
                        for k in range(len(payoff_pair))
                    )
                    and any(
                        payoff_pair[k] < other_payoff_pair[k]
                        for k in range(len(payoff_pair))
                    )
                    for other_i in range(len(self.game_matrix))
                    for other_j in range(len(self.game_matrix[other_i]))
                    for other_payoff_pair in [self.game_matrix[other_i][other_j]]
                ):
                    pareto_optimal_outcomes.append((i, j))

        if pareto_optimal_outcomes:
            for p_o in pareto_optimal_outcomes:
                print(
                    f"Pareto Optimal at index {p_o} with value {self.game_matrix[p_o[0]][p_o[1]]}"
                )
                return pareto_optimal_outcomes
        else:
            print("No Pareto Optimal found")
            return None

    def minimax_strategy(self, player):
        index = None
        minimax_regret_value = sys.maxsize

        # Strategy for player 1
        if player == 1:
            position = "row"
            # Look at each row
            for i in range(len(self.game_matrix)):
                # Calculate the maximum regret for each row based on column choice
                # Regret is the difference between the max value and other values
                # Max regret is the highest value of the regrets
                # If max is 4, then regret for outcome of 4 is 4 - 4 = 0
                # If outcome is 3, then regret is 4 - 3 = 1
                max_regret = max(
                    max(
                        self.game_matrix[row][col][0]
                        for row in range(len(self.game_matrix))
                    )
                    - self.game_matrix[i][col][0]
                    for col in range(len(self.game_matrix[i]))
                )
                # Compare max regret for this row against other rows, keep the lowest
                if max_regret < minimax_regret_value:
                    minimax_regret_value = max_regret
                    # store the row with the lowest max regret
                    index = i

        # Strategy for player 2
        else:
            position = "column"
            # Look at each column
            for j in range(len(self.game_matrix[0])):
                # Calculate the maximum regret for each row based on column choice
                max_regret = max(
                    max(
                        self.game_matrix[row][col][1]
                        for col in range(len(self.game_matrix[row]))
                    )
                    - self.game_matrix[row][j][1]
                    for row in range(len(self.game_matrix))
                )
                # Compare max regret for this column against other columns, keep the lowest
                if max_regret < minimax_regret_value:
                    minimax_regret_value = max_regret
                    # Store the column with the lowest max regret
                    index = j

        # Print the results
        if index:
            print(
                f"Player {player} at {position} {index} with a min maximum regret value of {minimax_regret_value}"
            )
            return index
        else:
            print(f"No single Minimax Strategy found for player {player}")
            return None

    def maximin_strategy(self, player):
        index = None
        maximin_value = -sys.maxsize

        # Strategy for player 1
        if player == 1:
            position = "row"
            # For each row, find the minimum value of player 1 at each column
            for i in range(len(self.game_matrix)):
                min_val = min(
                    self.game_matrix[i][j][0] for j in range(len(self.game_matrix[i]))
                )
                # If min value is greater than the current max, make it the new max
                if min_val > maximin_value:
                    maximin_value = min_val
                    # Store the row index with the highest minimum value
                    index = i
        # Strategy for player 2
        else:
            position = "column"
            # For each column, find the minimum value of player 2 at each row
            for j in range(len(self.game_matrix[0])):
                min_val = min(
                    self.game_matrix[i][j][1] for i in range(len(self.game_matrix))
                )
                # If min value is greater than the current max, make it the new max
                if min_val > maximin_value:
                    maximin_value = min_val
                    # Store the column index with the highest minimum value
                    index = j

        # Print the results
        if index:
            print(
                f"Player {player} at {position} {index} with max minimum value of {maximin_value}"
            )
            return index
        else:
            print(f"No single Maximin Strategy found for player {player}")
            return None

    def simulate_repetitive_game(self, strategies, repetitions):
        # initialize scores as 0
        total_scores = [0 for _ in strategies]

        # Reapet repetitions times
        for _ in range(repetitions):
            chosen_strategies = []

            # For each player, choose a strategy
            for strategy in strategies:
                # create random strategy based on length of options
                # Random for row (player 1)
                if strategy == "random1":
                    strategy = generate_random_mixed_strategy(self.row_length)
                # Random for column (player 2)
                elif strategy == "random2":
                    strategy = generate_random_mixed_strategy(self.column_length)

                # If strategy is a list, choose a random strategy based on the weights (example [.5, .5])
                if isinstance(strategy, list):
                    chosen_strategy = random.choices(
                        range(len(strategy)), weights=strategy
                    )[0]

                # If strategy is a number, choose that strategy (this includes strategies like minimax and maximin that return a single index)
                else:
                    chosen_strategy = strategy
                chosen_strategies.append(chosen_strategy)

            # calculate the payout for the chosen strategies
            payout = self.game_matrix[chosen_strategies[0]][chosen_strategies[1]]

            # add the scores
            for i in range(len(total_scores)):
                total_scores[i] += payout[i]

        # calculate the average scores
        average_scores = [score / repetitions for score in total_scores]
        print(f"Player 1 strategy: {strategies[0]}")
        print(f"Player 2 strategy: {strategies[1]}")
        print()
        print(f"P1 average score: {average_scores[0]}")
        print(f"P2 average score: {average_scores[1]}")


# help function to generate random mixed strategy based on length
def generate_random_mixed_strategy(length):
    probabilities = [random.random() for _ in range(length)]
    sum_probabilities = sum(probabilities)
    return [round(probability / sum_probabilities, 2) for probability in probabilities]


# Start the game
def start_game(file, repeated):
    # Create a normal form game
    game = NormalForm(file)
    # Create the game matrix
    game.create_game_matrix(file)

    # Find different strategy solutions
    print("\n" + "Weakly Dominated Strategies".center(60, "-"))
    game.solve_dominated_strategies(1, True)
    game.solve_dominated_strategies(2, True)

    print("\n" + "Strongly Dominated Strategies".center(60, "-"))
    game.solve_dominated_strategies(1, False)
    game.solve_dominated_strategies(2, False)

    print("\n" + "Nash Equilibrium".center(60, "-"))
    game.find_nash_equilibrium()

    print("\n" + "Pareto Optimal Outcomes".center(60, "-"))
    game.pareto_optimal_outcomes()

    print("\n" + "Minimax Strategy".center(60, "-"))
    minimax_1 = game.minimax_strategy(1)
    minimax_2 = game.minimax_strategy(2)

    print("\n" + "Maximin Strategy".center(60, "-"))
    maximin_1 = game.maximin_strategy(1)
    maximin_2 = game.maximin_strategy(2)

    # Simulate repetitive game

    if repeated:
        print("\n" + "Simulate Repetitive Game".center(60, "-"))

        repetitions = 1_000
        # random
        print("-" * 60)
        print(f"Completely Random Game ({repetitions} repetitions):")
        strategies = [
            "random1",
            "random2",
        ]
        game.simulate_repetitive_game(strategies, repetitions)

        # 1 vs. random
        print("-" * 60)
        print(f"Player 1 always 1 vs. Player 2 random ({repetitions} repetitions):")
        strategies = [
            [1],
            "random2",
        ]
        game.simulate_repetitive_game(strategies, repetitions)

        # 50/50 vs. minimax
        print("-" * 60)
        print(f"Player 1 50/50 vs. Player 2 minimax ({repetitions} repetitions):")
        strategies = [
            [0.5, 0.5],
            minimax_2,
        ]
        game.simulate_repetitive_game(strategies, repetitions)

        # maximin vs. maximin
        print("-" * 60)
        print(f"Player 1 maximin vs. Player 2 maximin ({repetitions} repetitions):")
        strategies = [
            maximin_1,
            maximin_2,
        ]
        game.simulate_repetitive_game(strategies, repetitions)


def main():
    files = [
        "prog3A.txt",
        # "prog3B.txt",
        # "prog3C.txt",
        # "prog3D.txt",
    ]
    for file in files:
        # Set to true for simulation
        start_game(file, True)


if __name__ == "__main__":
    main()
