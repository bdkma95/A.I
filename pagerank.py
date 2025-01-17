import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    # Number of pages in the corpus
    num_pages = len(corpus)
    
    # Initialize the distribution dictionary
    distribution = {p: 0 for p in corpus}

    # If the current page has links, calculate the probabilities
    if corpus[page]:
        links = corpus[page]
        num_links = len(links)
        
        # Probability of choosing a linked page
        for link in links:
            distribution[link] += damping_factor / num_links
        
        # Probability of choosing any page in the corpus
        for p in corpus:
            distribution[p] += (1 - damping_factor) / num_pages
    else:
        # If there are no outgoing links, treat it as having links to all pages
        for p in corpus:
            distribution[p] = 1 / num_pages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize ranks for each page
    ranks = {page: 0 for page in corpus}

    # Start with a random page
    current_page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        # Increment the count for the current page
        ranks[current_page] += 1
        
        # Get the transition probabilities for the current page
        distribution = transition_model(corpus, current_page, damping_factor)
        
        # Choose the next page based on the distribution
        current_page = random.choices(list(distribution.keys()), weights=distribution.values())[0]

    # Normalize ranks to sum to 1
    total_samples = sum(ranks.values())
    return {page: rank / total_samples for page, rank in ranks.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Parameters:
        corpus (dict): A dictionary mapping page names to sets of linked pages.
        damping_factor (float): The damping factor for random surfing.

    Returns:
        dict: A dictionary where keys are page names, and values are their estimated PageRank value.
    """
    num_pages = len(corpus)
    
    # Initialize ranks uniformly
    ranks = {page: 1 / num_pages for page in corpus}
    
    while True:
        new_ranks = {}
        for page in corpus:
            # Start with the base rank contribution
            new_rank = (1 - damping_factor) / num_pages
            
            # Sum contributions from all pages linking to this one
            for other_page in corpus:
                if page in corpus[other_page]:  # If other_page links to this page
                    new_rank += damping_factor * (ranks[other_page] / len(corpus[other_page]))
            
            new_ranks[page] = new_rank
        
        # Check for convergence (if ranks don't change significantly)
        if all(abs(new_ranks[page] - ranks[page]) < 0.001 for page in ranks):
            break
        
        ranks = new_ranks
    
    return ranks


if __name__ == "__main__":
    main()
