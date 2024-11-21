using Distributed
println("Starting processes...")
addprocs(25)  # Adjust the number of processors as needed

@everywhere begin
    ENV["GKSwstype"] = "png"
    using SharedArrays
    using ProgressMeter
    using HDF5
    using Statistics
    using Plots
    gr()
    using Plots: plot, heatmap, savefig, closeall, title!
    using StatsBase: percentile
end

# Function to generate and save recurrence plot
@everywhere function recurrence_plot(data::Vector{Float64}, perc_threshold::Float64)
    distance_matrix = abs.(data .- data')
    threshold = percentile(vec(distance_matrix), perc_threshold)
    recurrence_plot = distance_matrix .<= threshold
    return recurrence_plot
end

@everywhere function generate_and_save_recurrence_plot(pair::Tuple{Int, Int})
    try
        i, j = pair

        # # if image already exists, skip
        # save_path = joinpath("F:\\git\\network_miner\\temp\\recurrence", "$(lpad(i-1, 4, '0'))_$(lpad(j-1, 4, '0')).png")
        # if isfile(save_path)
        #     return
        # end

        if s_shared[i] && s_shared[j]
            label = "soz->soz"
        elseif s_shared[i] && !s_shared[j]
            label = "soz->noz"
        elseif !s_shared[i] && s_shared[j]
            label = "noz->soz"
        else
            label = "noz->noz"
        end

        data = vec(A_shared[:,i,j])
        # @show std(data)
        # threshold = std(data) * 0.2

        # @show threshold
        rec = recurrence_plot(data, 2.0)
        # display(rec[1:10, 1:10])
        # reverse!(rec, dims=2)
        # Plot and save
        p = heatmap(rec, color=:binary, yflip=false, colorbar=false,aspect_ratio=:equal,dpi=300,axis=false)
        string_i = uppercase(string(s_shared[i])[1])
        string_j = uppercase(string(s_shared[j])[1])
        title!(p, "$(lpad(i-1, 4, '0'))|$string_i, $(lpad(j-1, 4, '0'))|$string_j || $label")
        save_path = joinpath("F:\\git\\network_miner\\temp\\recurrence\\5%", "$(lpad(i-1, 4, '0'))_$(lpad(j-1, 4, '0')).png")
        savefig(p, save_path)
        closeall()
    catch e
        println("Error for pair $pair: $e")
    end
end

# Main function to load and process data
function main(input_file::String)
    println("Loading data from $input_file...")

    A_mats = h5read(input_file, "data/A_mats") # shaped (nwindows, channels, channels) ¯\_(ツ)_/¯. flipped from python
    A_mask = h5read(input_file, "data/A_mask")
    A_mats = A_mats[A_mask .== 1, :, :]
    A_mats = abs.(diff(A_mats; dims=1))
    # A_mats = diff(A_mats; dims=1)
    # println(size(A_mats))
    A_shared = SharedArray{Float64}(size(A_mats))
    A_shared .= A_mats
    @everywhere const A_shared = $A_shared

    soz = h5read(input_file, "metadata/patient_info/soz")
    s_shared = SharedArray{Bool}(size(soz))
    s_shared .= soz
    @everywhere const s_shared = $s_shared

    _, num_rows, num_cols = size(A_shared)
    pairs = [(i, j) for i in 1:num_rows for j in 1:num_cols]

    @showprogress pmap(generate_and_save_recurrence_plot, pairs)
    # for pair in pairs
    #     generate_and_save_recurrence_plot(pair)
    #     break
    # end
end


main("F:\\git\\eeg_prep\\processed_files\\064_000500_000500.hdf5")
