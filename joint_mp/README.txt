consensus_search is a Matlab class implementing the algorithms described in Finding and Exploiting Time Series Consensus Motifs. A reference example is also included. 

This class contains a single constructor, with the following signature
consensus_search(cell array containing all time series, subsequence length)

This sets up a few helper variables. In order to perform a basic search, construct a consensus_search object, then use obj.solve_opt(). This returns a struct with fields 

'nearest_neighbor_dists', the distance between one subsequence and its nearest neighbor in each time series
'nearest_neighbor_indices', the corresponding subsequence index with respect to each time series
'radius', the radius of a minimal encompassing hypersphere as described in Finding and Exploiting Time Series Consensus Motifs          


The k of P version also returns a struct with the same signature. Its function signature is
obj.solve_optimal_subset(minContain), where minContain is the smallest number of time series that must be used to calculate a match.


It's best to reuse a single object if subsequence length does not change, as a lot of things are cached at the object level. This is useful for the k of P variant if you wish to test different values of k against P time series.