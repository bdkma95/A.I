import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
        # Get the length of the variable
            variable_length = variable.length

            # Filter domain to ensure values have the correct length
            invalid_values = {word for word in self.domains [variable] if len(word) != variable_length}
        
            # Remove invalid values from the domain
            for word in invalid_values:
                self.domains[variable].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]

        # If x and y do not overlap, no need to revise
        if overlap is None:
            return revised

        # Get the indices of the overlapping characters
        x_index, y_index = overlap

        # Check all values in x's domain
        for word_x in self.domains[x].copy():
            # If there is no value in y's domain that matches the overlap, remove word_x
            if not any(word_x[x_index] == word_y[y_index] for   word_y in self.domains[y]):
                self.domains[x].remove(word_x)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Initialize the queue
        if arcs is None:
            queue = [
                (x, y) for x in self.crossword.variables
                for y in self.crossword.neighbors(x)
            ]
        else:
            queue = list(arcs)

        # Process arcs in the queue until no further revisions can be made
        while queue:
            x, y = queue.pop(0)

            # If the domain of x was revised, check if it became reduced to 1 value
            if self.revise(x, y):
                # If a domain becomes empty, return False
                if not self.domains[x]:
                    return False

                # If x's domain is reduced to one value, propagate the information
                if len(self.domains[x]) == 1:
                    # For each neighbor of x, add the (x, neighbor) arc to the queue
                    for z in self.crossword.neighbors(x) - {y}:
                        queue.append((z, x))

        return True
    
    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Check if all variables are assigned in the assignment dictionary
        return len(assignment) == len(self.crossword.variables) and all(assignment.values())

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Ensure all values are distinct
        if len(set(assignment.values())) != len(assignment.values()):
            return False

    # Check if each variable's value is the correct length
        for variable, word in assignment.items():
            if len(word) != variable.length:
                return False

    # Check for conflicts between neighboring variables
        for variable, word in assignment.items():
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:  # Only check assigned neighbors
                    overlap = self.crossword.overlaps[variable, neighbor]
                    if overlap is not None:
                        i, j = overlap
                        if word[i] != assignment[neighbor][j]:
                            return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of var, ordered   by the
        least-constraining values heuristic.

        Args:
            var (Variable): The variable whose domain is being considered.
            assignment (dict): The current assignment of variables to words.

        Returns:
            list: A list of values in the domain of var, ordered by the least-constraining values heuristic.
        """
        def count_conflicts(value):
            """
            Count the number of values ruled out for neighboring unassigned variables
            if var is assigned the given value.
            """
            conflicts = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap is not None:
                        i, j = overlap
                        # Count the number of conflicting values in neighbor's domain
                        for neighbor_value in self.domains  [neighbor]:
                            if neighbor_value[j] != value[i]:
                                conflicts += 1
            return conflicts

        # Return the domain values sorted by the number of conflicts they cause
        return sorted(self.domains[var], key=count_conflicts)


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Filter out variables that are already assigned
        unassigned = [var for var in self.crossword.variables   if var not in assignment]

        # Sort by the Minimum Remaining Values (MRV), then by Degree
        def heuristic(var):
            # Number of remaining values in the variable's domain
            remaining_values = len(self.domains[var])
            # Number of neighbors (degree) for the variable
            degree = len(self.crossword.neighbors(var))
            return (remaining_values, -degree)  # Degree is negated for descending sort

        # Return the variable with the best heuristic value
        return min(unassigned, key=heuristic)

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Base case: if assignment is complete, return it
        # Base case: if assignment is complete, return it
        if self.assignment_complete(assignment):
            return assignment

        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)

        # Try each value in the domain of the variable
        for value in self.order_domain_values(var, assignment):
            # Create a new assignment with the variable set to the value
            assignment[var] = value

            # Check if assignment remains consistent
            if self.consistent(assignment):
                # Recursively attempt to complete the assignment
                result = self.backtrack(assignment)
                if result is not None:
                    return result

            # If inconsistent, remove the assignment for this variable
            del assignment[var]

        # If no value is valid, return failure
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
