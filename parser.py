import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | VP
NP -> Det N | Det Adj N | N | NP PP
VP -> V | V NP | V PP | Adv VP | VP Adv
PP -> P NP
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.

    Parameters:
        sentence (str): The input sentence to preprocess.

    Returns:
        list: A list of lowercased words from the sentence.
    """
    # Tokenize the sentence into words
    from nltk.tokenize import word_tokenize
    words = word_tokenize(sentence)

    # Filter out non-alphabetic words and convert to lowercase
    words = [word.lower() for word in words if any(char.isalpha() for char in word)]

    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase (NP) chunks in the sentence tree.
    A noun phrase chunk is defined as a subtree whose label is "NP"
    and that does not itself contain other noun phrases.

    Parameters:
        tree (nltk.Tree): The syntax tree of the sentence.

    Returns:
        list: A list of nltk.Tree objects, each representing a noun phrase chunk.
    """
    # List to store all noun phrase chunks
    np_chunks = []

    # Iterate over all subtrees in the tree
    for subtree in tree.subtrees():
        # Check if the subtree is labeled as 'NP'
        if subtree.label() == 'NP':
            # Ensure it does not contain any smaller NP subtrees
            if not any(child.label() == 'NP' for child in subtree.subtrees(lambda t: t != subtree)):
                np_chunks.append(subtree)

    return np_chunks


if __name__ == "__main__":
    main()
