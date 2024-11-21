# Install necessary package
# using Pkg
# Pkg.add("Graphs")

using Base.Threads 
using HDF5
using Graphs


# Main function
function drop_lowest_connections_until_split(in::Array{Float64})
    n = size(in, 1)  # Assuming n is the number of nodes
    
    # Create edge list with absolute weights
    EdgeTuple = Tuple{Int, Int, Float64}  
    edges = EdgeTuple[]
    
    for i in 0:n-1
        for j in 0:n-1
            push!(edges, (i, j, abs(in[i+1,j+1])))
        end
    end
    
    # Sort edges by ascending absolute weight
    sort!(edges, by = e -> e[3])
    # display(edges)
    # Create directed graph
    g = DiGraph(n)
    for e in edges
        add_edge!(g, e[1], e[2])
    end
    if !is_strongly_connected(g)
        println("Graph is not strongly connected.")
        return in
    end 
    
    # Iterate through sorted edges and remove them
    for e in edges
        rem_edge!(g, e[1], e[2])
        if !is_strongly_connected(g)
            # If disconnects, add it back and stop
            add_edge!(g, e[1], e[2])
            println("Stopping removal. Removing edge ($(e[1]), $(e[2])) would disconnect the graph.")
            break
        else
            # If still connected, set adjacency matrix entry to zero
            in[e[1]+1, e[2]+1] = 0.0
        end
    end
    
    return in
end

input_file = "f:\\git\\eeg_prep\\me\\processed_files\\001_000500_000500.hdf5"
input_file = "/media/dan/Data/git/eeg_prep/processed_files/001_000500_000500.hdf5"
output_file = "julia_network_reduction_test1.hdf5"
rm(output_file, force=true)
A_mats = h5read(input_file, "data/A_mats")
A_mats = permutedims(A_mats, (1, 3, 2)) # order the same way as in python

diffs = A_mats[2:end, :, :] - A_mats[1:end-1, :, :]
# display(size(diffs))
a = diffs[1, :, :]
# display(a[1:5, 1:5])
out = drop_lowest_connections_until_split(a)

h5write(output_file, "data/diffs", diffs)

# # Parallel processing using threads
# results = Array{Array{Float64}}(undef, size(diffs, 1))

# @threads for i in axes(diffs, 1)
#     a = diffs[i, :, :]
#     results[i] = drop_lowest_connections_until_split(a)
# end

# # Assuming all arrays in 'results' are of the same size
# result_array = cat(results...; dims=3)
# # Save the results array
# h5write(output_file, "data/results", result_array)