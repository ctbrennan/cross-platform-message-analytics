A program written in Python, using Sci-kit learn and numpy libraries. 
Parses iMessage database files, Android SMS archives, and Facebook message archives and puts them in various data structures, combining communications with the same people in chronological order across platforms.
Automatically resolves differences in how people are named using minimum edit distances and cosine similarity between TFIDF vectors.
Includes several analytic tools. Creates line graphs of frequency of communication with your closest friends and/or their average sentiment using matplotlib and generates wordclouds for any individual. 
Creates 2 and 3-dimensional scatterplots where each node represents a friend and their proximity represents the degree of similarity of their word usage, as computed by using principal component decomposition on higher-dimensional TFIDF vectors.
Performs a couple other interesting analytic tasks, with more to come.

Next up: supporting WhatsApp messages