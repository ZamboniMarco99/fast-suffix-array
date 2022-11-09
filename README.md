# Fast suffix array

This repository implements in python a way to build the suffix array of a string in O(1) time thanks to the Skew/DC3 algorithm.

On top of it this is a fast implementation that thanks to numpy and numba can run several times faster than the pure python one.

## References

[Paper describing the algorithm](https://www.cs.helsinki.fi/u/tpkarkka/publications/jacm05-revised.pdf)

Thanks to professor Mailund of Aarhus University for his [blog post](https://mailund.dk/posts/skew-python-go/) where they give the implementation that I used for reference
