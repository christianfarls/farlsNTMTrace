import csv
import sys
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict

# Transition dataclass allows for easy access of transition data for the dictionary
@dataclass
class Transition:
    current_state: str
    read_symbol: str
    next_state: str
    write_symbol: str
    direction: str

# Configuration class allows for easier storage of configs for tree building
class Configuration:
    def __init__(self, left: str, state: str, head: str, right: str):
        self.left = left
        self.state = state
        self.head = head
        self.right = right

    # Converts a config into a string
    def __str__(self) -> str:
        return f"{self.left}, {self.state}, {self.head}{self.right}"

    # Defines when two configs are considered equal
    def __eq__(self, other) -> bool:
        if not isinstance(other, Configuration):
            return False
        return (self.left == other.left and
                self.state == other.state and
                self.head == other.head and
                self.right == other.right)

# NTM class for all machine and traversal management
class NTM:
    def __init__(self, filename: str):
        self.transitions: dict = defaultdict(list)
        self.total_transitions = 0
        self.load_machine(filename)

    # Loads machine with initial data
    def load_machine(self, filename: str) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            # Parses header for machine information
            self.name = next(reader)[0] # Machine name
            self.states = set(next(reader)[0].split(',')) # All states
            self.input_alphabet = set(next(reader)[0].split(',')) # Input alphabet
            self.tape_alphabet = set(next(reader)[0].split(',')) # Tape alphabet
            self.start_state = next(reader)[0] # Start state
            self.accept_state = next(reader)[0] # Accept state
            self.reject_state = next(reader)[0] # Reject state

            # Reads transitions following header
            for row in reader:
                if len(row) == 5:
                    curr_state, read_sym, next_state, write_sym, direction = row
                    self.transitions[(curr_state, read_sym)].append(
                        Transition(curr_state, read_sym, next_state, write_sym, direction)
                    )

    # Gets all next possible configs from current config
    def get_next_configurations(self, config: Configuration) -> List[Configuration]:
        next_configs = []

        # Get possible transitions for current state and head symbol
        possible_transitions = self.transitions.get((config.state, config.head), [])

        # If no transitions defined, transition to reject state
        if not possible_transitions and config.state != self.reject_state:
            return [Configuration(config.left, self.reject_state, config.head, config.right)]

        # Iterates through transitions and captures config data
        for trans in possible_transitions:
            new_left = config.left
            new_right = config.right
            new_head = trans.write_symbol

            if trans.direction == 'L':
                if new_left:
                    new_head = new_left[-1]
                    new_left = new_left[:-1]
                    new_right = trans.write_symbol + new_right
                else:
                    new_head = '_'
                    new_right = trans.write_symbol + new_right

            elif trans.direction == 'R':
                if new_right:
                    new_left = new_left + trans.write_symbol
                    new_head = new_right[0]
                    new_right = new_right[1:]
                else:
                    new_left = new_left + trans.write_symbol
                    new_head = '_'

            next_configs.append(Configuration(new_left, trans.next_state, new_head, new_right))

        return next_configs

    # Calculates the degree of nondeterminism as instructed
    def calculate_nondeterminism(self, config_tree: List[List[Configuration]]) -> float:
        total_transitions = 0
        total_nonleaves = 0

        # Iterates through each tree level and counts transitions and non-leaves
        for level in config_tree:
            nonleaves = sum(1 for config in level
                            if config.state not in [self.accept_state, self.reject_state])
            if nonleaves > 0:
                transitions = sum(len(self.transitions.get((config.state, config.head), []))
                                  for config in level)
                total_transitions += transitions
                total_nonleaves += nonleaves

        return total_transitions / total_nonleaves if total_nonleaves > 0 else 1.0

    # Simulates NTM
    def simulate(self, input_string: str, verbose: bool = False, max_depth: int = 100) -> Tuple[
        bool, int, List[List[Configuration]], Optional[List[Configuration]]]:

        # Initialize configuration tree
        config_tree: List[List[Configuration]] = []
        head = input_string[0] if input_string else '_'
        right = input_string[1:] if len(input_string) > 1 else ""
        initial_config = Configuration("", self.start_state, head, right)
        config_tree.append([initial_config])

        # Computes a BFS
        for depth in range(max_depth):
            if not config_tree[depth]:
                return False, depth, config_tree, None

            current_level = []

            # Iterates through configurations in the = tree
            for config in config_tree[depth]:
                # Checks for accept state
                if config.state == self.accept_state:
                    if verbose:
                        print(f"\nFound accepting configuration at level {depth}:")
                        print(f"  {str(config)}")
                    return True, depth, config_tree, self.reconstruct_path(config_tree, depth, config)

                # If not in reject state, continue search
                if config.state != self.reject_state:
                    next_configs = self.get_next_configurations(config)
                    current_level.extend(next_configs)

            # If all paths rejected, return reject state
            if not current_level:
                return False, depth, config_tree, None

            config_tree.append(current_level)

        return False, max_depth, config_tree, None

    # Reconstructs the path that lead to accept state
    def reconstruct_path(self, tree: List[List[Configuration]], depth: int,
                         final_config: Configuration) -> List[Configuration]:
        path = [final_config]
        current_config = final_config
        current_level = depth

        # Work backwards through the tree to find the path
        while current_level > 0:
            prev_level = tree[current_level - 1]
            # Find configuration in previous level that could lead to current config
            for prev_config in prev_level:
                next_configs = self.get_next_configurations(prev_config)
                if any(config == current_config for config in next_configs):
                    path.append(prev_config)
                    current_config = prev_config
                    break
            current_level -= 1

        # Reverse list of final -> start and return
        return list(reversed(path))

# Helps parse user arguments for different options
def parse_args():
    parser = argparse.ArgumentParser(description='NTM Simulator')
    parser.add_argument('tm_file', help='TM definition file (CSV format)')
    parser.add_argument('input_string', help='Input string to process')
    parser.add_argument('-d', '--max-depth', type=int, default=100,
                        help='Maximum depth for configuration tree (default: 100)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file for results')
    return parser.parse_args()

# Writes output to both stdout and file (if requested by user)
def write_output(file, text):
    print(text)
    if file:
        print(text, file=file)

# Create output dictionary if it doesn't exit
def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Get input file by entering code/ directory
def get_project_input_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, 'data', filename)

# Get output file in output/ directory
def get_project_output_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, 'output', filename)

def main():
    args = parse_args()

    outfile = None
    if args.output:
        try:
            # Use the project-level output directory
            output_path = get_project_output_path(args.output)
            ensure_directory_exists(output_path)
            outfile = open(output_path, 'w')
            print(f"Writing output to: {output_path}")
        except OSError as e:
            print(f"Error creating output file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Use the project-level data directory for input files
        input_path = get_project_input_path(args.tm_file)
        try:
            ntm = NTM(input_path)
        except FileNotFoundError:
            print(f"Error: Could not find TM definition file: {input_path}", file=sys.stderr)
            sys.exit(1)

        accepted, depth, config_tree, path = ntm.simulate(args.input_string, args.verbose, args.max_depth)

        # Print initial information
        write_output(outfile, f"\nMachine: {ntm.name}")
        write_output(outfile, f"Input: {args.input_string}")
        write_output(outfile, f"Tree depth: {depth}")

        # Calculate and print nondeterminism
        nondeterminism = ntm.calculate_nondeterminism(config_tree)
        write_output(outfile, f"Degree of nondeterminism: {nondeterminism:.2f}\n")

        # If verbose output requested, print entire config tree at each level
        if args.verbose:
            write_output(outfile, "\nConfiguration Tree:")
            for level, configs in enumerate(config_tree):
                write_output(outfile, f"Level {level}:")
                for conf in configs:
                    write_output(outfile, f"  {str(conf)}")
                write_output(outfile, "")

        # If the string is accepted, print acceptance path
        if accepted:
            write_output(outfile, f"String accepted in {depth} steps")
            if path:
                write_output(outfile, "\nAccepting path with transitions:")
                for i in range(len(path) - 1):
                    current = path[i]
                    next_config = path[i + 1]
                    # Find the transition that was taken
                    transitions = ntm.transitions.get((current.state, current.head), [])
                    transition = next(t for t in transitions
                                    if any(nc == next_config
                                         for nc in ntm.get_next_configurations(current)))
                    write_output(outfile, f"  {str(current)}")
                    write_output(outfile,
                                f"  ↓ ({transition.current_state},{transition.read_symbol}→{transition.write_symbol},{transition.direction})")
                write_output(outfile, f"  {str(path[-1])}")  # Print final configuration
        elif depth == args.max_depth:
            write_output(outfile, f"Execution stopped after {args.max_depth} steps")
        else:
            write_output(outfile, f"String rejected in {depth} steps")

    finally:
        if outfile:
            outfile.close()

if __name__ == "__main__":
    main()