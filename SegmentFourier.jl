using Images, ImageView, ImageSegmentation, Plots, FFTW, GeoArrays, ImageFiltering, Statistics, StatsBase, Distributions, Interpolations, Dates, Turing, Random, Glob, DataFrames

# Function for cropping image for dimensional consistency
function crop_center(img::AbstractArray, crop_size::Tuple{Int, Int})
    img_height, img_width = size(img)
    crop_height, crop_width = crop_size
    center_y, center_x = div(img_height, 2), div(img_width, 2)
    
    top = max(1, center_y - div(crop_height, 2))
    bottom = min(img_height, top + crop_height - 1)
    left = max(1, center_x - div(crop_width, 2))
    right = min(img_width, left + crop_width - 1)
    
    return img[top:bottom, left:right]
end

#Function to preview imagery and select coordinates
function preview(file::Union{String, AbstractArray}; crop_size::Union{Nothing, Tuple{Int, Int}}=nothing)
    if typeof(file) == String
        img = load(file)
    else
        img = file
    end

    if crop_size != nothing
        img = crop_center(img, crop_size)
    end
    imshow(img)
end

function NDVI(chip; source::String="Sentinel", threshold::Union{Float64, Nothing}=nothing)
    if source == "Sentinel"
        chip_red = (chip)[:, :, 1]   # Red channel
        chip_nir = (chip)[:, :, 4]   # NIR channel
    elseif source == "Planet" || source == "Landsat"
        chip_red = (chip)[:, :, 3]   # Red channel
        chip_nir = (chip)[:, :, 4]   # NIR channel
    else
        println("The data source string must be either 'Sentinel', 'Planet', or 'Landsat'.")
    end

    ndvi_chip = (chip_nir .- chip_red) ./ (chip_nir .+ chip_red)

    if threshold != nothing
        threshold = percentile(ndvi_chip[:], threshold)
        ndvi_chip = map(x -> x <= threshold ? 1 : 0, ndvi_chip)
    end

    return ndvi_chip
end

function TrueColor(path; crop_size=nothing, source::String="Sentinel")
    if typeof(path) == String
        chip = GeoArrays.read(path)
    else
        chip = path
    end

    if crop_size != nothing
        chip = crop_center(chip, crop_size)
    end

    if source=="Sentinel"
        chip_red = chip[:, :, 1]   # Red channel
        chip_green = chip[:, :, 2]   # Green channel
        chip_blue = chip[:, :, 3]   # Blue channel
    elseif source=="Planet" || source=="Landsat"
        chip_red = chip[:, :, 3]   # Red channel
        chip_green = chip[:, :, 2]   # Green channel
        chip_blue = chip[:, :, 1]   # Blue Channel
    else
        println("The data source must be a string of either 'Sentinel' or 'Planet'.")
    end

    red_band_normalized = chip_red / maximum(chip_red)
    green_band_normalized = chip_green / maximum(chip_green)
    blue_band_normalized = chip_blue / maximum(chip_blue)

    rgb_image = colorview(RGB, red_band_normalized, green_band_normalized, blue_band_normalized)
    color_plot = Plots.plot(rgb_image', axis=false, grid=false)
    return color_plot, rgb_image
end

function display_segments(segments)
    segment = map(i -> segment_mean(segments, i), labels_map(segments))
    plot = Plots.plot(segment, framestyle=:none)
    display(plot)
end

# Function to get pixel count of segmented area
function count_pixels(object::Union{String, AbstractArray}, Seed1::Tuple{Int64,Int64}, Seed2::Tuple{Int64,Int64}; Seed3::Union{Nothing, Tuple{Int64,Int64}} = nothing, Seed4::Union{Nothing, Tuple{Int64,Int64}} = nothing, Display::Bool = false, crop_size::Union{Nothing, Tuple{Int, Int}}=nothing, mods::Union{Nothing, Vector{Tuple{Real, Real, Vararg{Float64}}}}=nothing, water_mask::Union{Nothing, BitMatrix}=nothing, ndvi_threshold::Union{Nothing, Float64}=nothing, verbose::Bool = false)

    # Load image from file path if the input is a string
    if typeof(object) == String
        file_path = object
        if endswith(file_path, ".png")
            img = load(file_path)
        else
            println("File is not a .png: $file_path")
            return nothing
        end
    else
        img = object
    end

    if crop_size != nothing
        img = crop_center(img, crop_size)
    end

    if mods != nothing
        img = draw_line(img, mods)
    end

    if water_mask != nothing
        img = img .* water_mask
    end

    if ndvi_threshold != nothing
        img = img .< ndvi_threshold
    end
        
    if Seed3 == nothing && Seed4 == nothing
        seeds = [(CartesianIndex(Seed1),1), (CartesianIndex(Seed2),2)]
    elseif Seed3 != nothing && Seed4 == nothing
        seeds = [(CartesianIndex(Seed1),1), (CartesianIndex(Seed2),2), (CartesianIndex(Seed3),3)]
    else
        seeds = [(CartesianIndex(Seed1),1), (CartesianIndex(Seed2),2), (CartesianIndex(Seed3),3), (CartesianIndex(Seed4),4)]
    end
        
    segments = seeded_region_growing(img, seeds)    
    pixel_dict = segment_pixel_count(segments)
    pixel_count = pixel_dict[:1]
        
    if Display
        display_segments(segments)
    end

    if verbose
        println("The segemented region contains $pixel_count pixels.")
    end

    return pixel_count, segments
end


# Function to get cetroid and egde pixels from image segmentation result
function Prepare(matrix)
    # Convert matrix to binary: 1 if 1, otherwise 0
    binary_matrix = map(x -> x == 1 ? 1 : 0, matrix)
    
    # Matrix dimensions
    rows, cols = size(binary_matrix)
    
    # Find all indices of 1s in the binary matrix
    indices = findall(x -> x == 1, binary_matrix)
    
    # Separate row and column indices for centroid calculation
    row_indices = [i[1] for i in indices]
    col_indices = [i[2] for i in indices]
    
    # Calculate the centroid of the 1s
    centroid_y = mean(row_indices)
    centroid_x = mean(col_indices)
    centroid = (centroid_x, centroid_y)

    # Function to check if an adjacent cell is 0
    function EdgeExtraction(binary_matrix, r, c)
        # Check bounds and adjacent cells
        if r > 1 && binary_matrix[r - 1, c] == 0
            return true
        elseif r < rows && binary_matrix[r + 1, c] == 0
            return true
        elseif c > 1 && binary_matrix[r, c - 1] == 0
            return true
        elseif c < cols && binary_matrix[r, c + 1] == 0
            return true
        end
        return false
    end

    # Collect edge coordinates where 1s are adjacent to 0s
    edges = [(Tuple(i)...,) for i in indices if EdgeExtraction(binary_matrix, Tuple(i)...)]
    
    # Return both the centroid and edges
    return centroid, edges
end

# Function to display edges from image segmentation and centroid
function PlotEdge(matrix, edges, centroid)
    # Extract x and y coordinates of edges
    x_edges = [x for (x, _) in edges]
    y_edges = [y for (_, y) in edges]

    # Plot the matrix background
    heatmap(matrix', c=:greys, colorbar=false, xlims=(1, size(matrix, 2)), ylims=(1, size(matrix, 1)), ylabel="Pixel Rows", xlabel="Pixel Columns")

    # Overlay edge pixels with scatter plot
    scatter!(x_edges, y_edges, color=:red, label="Edge Pixels", yflip=true, markersize=8)

    # Add number labels to each marker
    for i in 1:length(edges)
        Plots.annotate!(x_edges[i], y_edges[i], text(string(i), :black, 5))
    end

    # Plot centroid
    scatter!([centroid[2]], [centroid[1]], color=:goldenrod, marker=:cross, label="Centroid")
end

# Function to find the first edge pixel intersected by rays at specified angles
function RaySelect(centroid, edges; angle_range=0:10:350)
    intersecting_edges = []
    
    for angle in angle_range
        # Convert angle to radians
        rad = angle * π / 180
        
        # Define ray direction from the centroid
        dx = cos(rad)
        dy = sin(rad)
        
        # Incrementally extend the ray and check for intersections
        for step in 1:100  # Choose a maximum step size
            x = centroid[1] + step * dx
            y = centroid[2] + step * dy
            
            # Round to nearest integer for pixel matching
            ix, iy = round(Int, x), round(Int, y)
            
            # Check if the current pixel (ix, iy) is an edge pixel
            if (ix, iy) in edges
                push!(intersecting_edges, (ix, iy))
                break
            end
        end
    end
    
    return intersecting_edges
end

# Function to detect edges on chip using edge detection (Different from edge extraction approach above)
function EdgeDetection(chip; sigma::Int=5, vizual::Bool=false)
    edge = canny(chip, (Percentile(80), Percentile(20)), sigma)
    edge = dilate(edge)

    labeled_edges = label_components(edge)

    # Count the number of pixels in each component and find the largest one
    component_sizes = countmap(labeled_edges)
    delete!(component_sizes, 0)

    _, label = findmax(component_sizes)

    # Step 4: Isolate the main edge
    main_edge = labeled_edges .== label

    main_edge = erode(main_edge) 

    if vizual
        main_edge = permutedims(main_edge, (1, 2)) # Adjust orientation for display
        viz = heatmap(main_edge, yflip=true)
        return main_edge, viz
    else
        return main_edge
    end
end

function Oreientation(chip1, chip2; sigma::Float64=1.0, threshold::Float64 = 0.01)
    edge1 = EdgeDetection(chip1)
    edge2 = EdgeDetection(chip2)

    edges = edge1 .+ edge2
    chip_diff = chip2 - chip1

    # Calculate gradients on the smoothed first image
    smoothed_chip1 = imfilter(chip1, Kernel.gaussian(sigma))
    grad_x, grad_y = imgradients(smoothed_chip1, KernelFactors.sobel)
    Gdir = atan.(grad_y, grad_x)  # Gradient direction

    # Create a mask for significant changes
    v = (edges .> 0) .& (abs.(chip_diff) .> threshold) .& (Gdir .!= 0)
        
    # Calculate average change along the edge with significant differences
    avg_change = mean(chip_diff[v])

    # Extract gradient directions at these points
    angles_all = rad2deg.(Gdir[v])

    # Fliter NaN values
    angles_all = angles_all[.!isnan.(angles_all)]

    # Compute histogram and edges
    histogram = fit(Histogram, angles_all, nbins=20)
    hist_edges = histogram.edges[1]
    hist_counts = histogram.weights

    # Calculate bin centers
    centers = [(hist_edges[i] + hist_edges[i+1]) / 2 for i in 1:(length(hist_edges)-1)]

    # Find the bin with the maximum count
    _, Itemp = findmax(hist_counts)
    angle_shift = centers[Itemp]

    # Wrap the angles to [-90, 90] around the main angle
    angle_wrapped = atan.(tan.(deg2rad.(angles_all .- angle_shift))) .|> rad2deg

    # Fit a normal distribution to the wrapped angles
    if sum(.!isnan.(angle_wrapped)) > 1
        a = fit(Normal, angle_wrapped)
        bank_orientation = [mean(a) + angle_shift, std(a)]
    else
        bank_orientation = [NaN, NaN]
    end
    return bank_orientation, v
end

function xcorr2_fft(a::AbstractMatrix, b::AbstractMatrix)
    # If only one argument is provided, autocorrelate
    if b === nothing
        b = a
    end

    # Matrix dimensions
    adim = size(a)
    bdim = size(b)

    # Cross-correlation output dimensions
    cdim = (adim[1] + bdim[1] - 1, adim[2] + bdim[2] - 1)

    # Pad `a` and `b` to the cross-correlation dimension
    apad = zeros(ComplexF64, cdim)
    bpad = zeros(ComplexF64, cdim)

    # Assign `a` to the top-left corner of `apad`
    apad[1:adim[1], 1:adim[2]] .= a

    # Assign the flipped `b` to the top-left corner of `bpad`
    bpad[1:bdim[1], 1:bdim[2]] .= reverse(reverse(b, dims=1), dims=2)

    # Compute the 2D FFT of both padded matrices
    fft_a = fft(apad)
    fft_b = fft(bpad)

    # Multiply the FFT of `a` by the FFT of `b`, and take the inverse FFT
    correlation_fft = fft_a .* fft_b
    c = real(ifft(correlation_fft))

    # Step 1: Find the peak location in the correlation matrix
    peak_index = argmax(c)
    
    # Convert the linear index to row and column (y, x) coordinates
    peak_coords = CartesianIndices(size(c))[peak_index]
    peak_y, peak_x = Tuple(peak_coords)

    # Step 2: Calculate the center of the matrix
    center_y = size(c, 1) / 2
    center_x = size(c, 2) / 2

    # Step 3: Calculate the displacement from the center
    displacement_y = peak_y - center_y
    displacement_x = peak_x - center_x

    # Step 4: Calculate the magnitude of the displacement
    displacement_magnitude = sqrt(displacement_x^2 + displacement_y^2)

    return c, displacement_magnitude
end

function Displacement(chip1, chip2, correlation; length::Int=10, num_angles::Int=5)
    center = (size(correlation, 1) / 2, size(correlation, 2) / 2)

    angle, mergedEdge = Oreientation(chip1, chip2)
    dTheta = range(-angle[2], angle[2], length=num_angles)

    itp = interpolate(correlation, BSpline(Cubic(Line(OnGrid()))), OnGrid())
    cxy_all = []

    for θ in dTheta
        angle_rad = deg2rad(angle[1] + θ)
        
        # Sample both forward and backward along the angle
        line_points = [(center[1] + t * cos(angle_rad), center[2] + t * sin(angle_rad)) for t in range(-length, stop=length, step=0.02)]
        
        valid_points = filter(p -> 1 <= p[1] <= size(correlation, 1) && 1 <= p[2] <= size(correlation, 2), line_points)
        
        sampled_values = [itp(p[1], p[2]) for p in valid_points]
        
        max_index = argmax(sampled_values)
        max_point = valid_points[max_index]
        
        push!(cxy_all, max_point)
    end

    dx = median([p[1] - center[1] for p in cxy_all])
    dy = median([p[2] - center[2] for p in cxy_all])

    return dx, dy, mergedEdge
end

# Function to extract a 32x32 chip centered on (cx, cy) from the matrix
function extract_chip(img, cx::Int, cy::Int, chip_size::Int=32)
    nrows, ncols = size(img)
    half_chip = div(chip_size, 2)

    # Calculate the bounds of the chip
    x_start = max(cx - half_chip, 1)
    x_end = min(cx + half_chip - 1, ncols)
    y_start = max(cy - half_chip, 1)
    y_end = min(cy + half_chip - 1, nrows)

    # Extract the chip
    chip = img[y_start:y_end, x_start:x_end]
    return chip
end

function InspectChip(path1::String, path2::String, Bounds; sigma=5, source::String="Sentinel", threshold::Union{Float64, Nothing}=nothing, crop_size::Union{Tuple,Nothing}=nothing)
    img1 = GeoArrays.read(path1);
    img2 = GeoArrays.read(path2);

    if crop_size !== nothing
        img1 = crop_center(img1, crop_size)
        img2 = crop_center(img2, crop_size)
    end

    cx, cy = round(Int, Bounds[1]), round(Int, Bounds[2])

    chip1 = extract_chip(img1, cx, cy);
    chip2 = extract_chip(img2, cx, cy);

    A, _ = TrueColor(chip1, crop_size=crop_size, source=source)
    B, _ = TrueColor(chip2, crop_size=crop_size, source=source)

    chip1_ndvi = NDVI(chip1, threshold=threshold, source=source);
    chip2_ndvi = NDVI(chip2, threshold=threshold, source=source);

    C = Plots.heatmap(permutedims(chip1_ndvi, (1, 2)), yflip=true, clims=(0,1))
    D = Plots.heatmap(permutedims(chip2_ndvi, (1, 2)), yflip=true, clims=(0,1))

    _, E = EdgeDetection(chip1_ndvi, sigma=sigma, vizual=true)
    _, F = EdgeDetection(chip2_ndvi, sigma=sigma, vizual=true)

    E = plot(C,D,E,F, layout=(2,2))
end

function Correlate(path1::String, path2::String, Bounds; clipping::Int=4, sigma::Float64=1.0, crop_size::Union{Tuple,Nothing}=nothing, source::String="Sentinel", threshold::Union{Float64, Nothing}=nothing)
    img1 = GeoArrays.read(path1);
    img2 = GeoArrays.read(path2);

    if crop_size !== nothing
        img1 = crop_center(img1, crop_size)
        img2 = crop_center(img2, crop_size)
    end

    cx, cy = round(Int, Bounds[1]), round(Int, Bounds[2])

    chip1 = extract_chip(img1, cx, cy);
    chip2 = extract_chip(img2, cx, cy);

    chip1_ndvi = NDVI(chip1, source=source, threshold=threshold);
    chip2_ndvi = NDVI(chip2, source=source, threshold=threshold);

    chip1_matrix = Matrix(chip1_ndvi);
    chip2_matrix = Matrix(chip2_ndvi);

    chip1_grad = imgradients(chip1_matrix, KernelFactors.sobel)[1]
    chip2_grad = imgradients(chip2_matrix, KernelFactors.sobel)[1]

    chip1_smoothed = imfilter(chip1_grad, Kernel.gaussian(sigma))
    chip2_smoothed = imfilter(chip2_grad, Kernel.gaussian(sigma))

    chip1_clipped = chip1_smoothed[(1+clipping):(end-clipping), (1+clipping):(end-clipping)]

    correlation, displacement = xcorr2_fft(chip1_clipped, chip2_smoothed)

    correlation_surface = Plots.surface(correlation, xlabel = "X", ylabel = "Y", zlabel = "Magnitude", color = :viridis)

    dx, dy, mergedEdge = Displacement(chip1_ndvi, chip2_ndvi, correlation)

    mergedEdge = reverse(Matrix(mergedEdge),dims=1)
    map = Plots.heatmap(mergedEdge, axis=false)
    Plots.quiver!(map, [size(mergedEdge,1)/2, size(mergedEdge,1)/2], [size(mergedEdge,1)/2, size(mergedEdge,1)/2], quiver=((0,dx), (0,-dy)), color="red")
    
    displacement_mag = sqrt(dx^2 + dy^2)
    
    return correlation_surface, map, displacement_mag, dx, dy
end

function ProcessStack(Images::Vector{String}, Bounds; crop_size::Union{Tuple,Nothing}=nothing, source::String="Sentinel", threshold::Union{Float64, Nothing}=nothing)
    dx_results = zeros(length(Images),length(Images))
    dy_results = zeros(length(Images),length(Images))
    results = zeros(length(Images),length(Images))

    for i in 1:(length(Images))
        for j in 1:length(Images)
            if i != j
                try
                    _,_,disp,dx,dy = Correlate(Images[i],Images[j],Bounds, crop_size=crop_size, source=source, threshold=threshold)
                    dx_results[i,j] = dx
                    dy_results[i,j] = -dy
                    if i > j
                        results[i,j] = -disp
                    else
                        results[i,j] = disp
                    end
                catch e
                    if isa(e, DimensionMismatch)
                        println("Skipping image due to DimensionMismatch.")
                        continue 
                    else
                        rethrow(e)  # Re-throw if it is a different error
                    end
                end
            else
                results[i,j] = 0.0
            end
        end
    end
    step_results = hcat(dx_results, dy_results)
    return results, step_results
end

function ExtractDate(Paths)
    date_vec = []
    for i in 1:length(Paths)
        base = basename(Paths[i])
        if length(base) > 14
            date_format = DateFormat("yyyy-mm-dd")
            date_str = base[11:20]
            date = Date(date_str, date_format)
            push!(date_vec, date)
        elseif length(base) == 14
            date_format = DateFormat("mm-dd-yyyy")
            date_str = base[1:10]
            date = Date(date_str, date_format)
            push!(date_vec, date)
        end
    end
    return date_vec
end

function Bracket_Stack(matrix, bracket_step, dates)
    # Select every `bracket_step` element from the vector
    bracketed = matrix[:, 1:bracket_step:end]
    brack_dates = dates[1:bracket_step:end]

    # Compute the cumulative sum of the bracketed vector
    row_averaged = vec(mean(bracketed, dims=1))
    
    return row_averaged, brack_dates
end

function MonotonicIncreasing(V, dates)
    # Start with the first element in the result
    monotonic_vector = [V[1]]
    monotonic_dates = [dates[1]]
    
    # Iterate over the vector
    for i in 2:length(V)
        # If the current element is greater than or equal to the last element in the monotonic vector, add it
        if V[i] >= monotonic_vector[end]
            push!(monotonic_vector, V[i])
            push!(monotonic_dates, dates[i])
        end
    end
    
    return monotonic_vector, monotonic_dates
end

# Function to get all GeoTIFF files in the folder
function ImagePaths(folder; type::String="*.tif")
    stacked = glob(type, folder)  
end


function Preprocess(Paths, index, Seed1::Tuple, Seed2::Tuple, angle_range; Display=false, crop_size::Union{Tuple,Nothing}=nothing)
    image = Paths[index]
    ColorPlot, ColorImg = TrueColor(image, crop_size=crop_size)

    n, p = count_pixels(ColorImg, Seed1, Seed2, crop_size=crop_size)
    centroid, edges = Prepare(p.image_indexmap)
    println("Mask Centroid: $(centroid)")

    SelectedEdges = RaySelect(centroid, edges, angle_range=angle_range)

    if Display
        edgeplot = PlotEdge(p.image_indexmap, SelectedEdges, centroid)
        display(edgeplot)
        return SelectedEdges, edgeplot
    end

    return SelectedEdges
end

function AnalyzeSlump(Paths::Vector{String}, Bounds; BracketStep::Int=2, crop_size::Union{Tuple,Nothing}=nothing, source::String="Sentinel", threshold::Union{Float64, Nothing}=nothing)
    ImageDates = ExtractDate(Paths)
    TimeSeriesList = []

    for i in eachindex(Bounds)
        correlation_matrix, step_results = ProcessStack(Paths, Bounds[i], source=source, threshold=threshold, crop_size=crop_size)
        Processed, BracketedDates = Bracket_Stack(correlation_matrix, BracketStep, ImageDates)
        MonotonicData, MonotonicDates = MonotonicIncreasing(Processed, BracketedDates)
        index = fill(i, length(MonotonicData))
        Data = hcat(MonotonicData,MonotonicDates,index)
        push!(TimeSeriesList, Data)
    end

    return TimeSeriesList
end

function TimeSeriesPlot(data::Union{Vector{Any}, Matrix{Any}}, points::Int)
    if typeof(data) == Vector{Any}
        if points == 1
            subdata = data[1]
            MonotonicData = subdata[:,1]
            MonotonicDates = subdata[:,2]
            LabelPoint = subdata[1,3]
            outplot = Plots.scatter(MonotonicDates,MonotonicData .+ abs(minimum(MonotonicData)),xlabel="Date",ylabel="Monotonic Cumulative Displacment", label="Point Index: $(LabelPoint)")
        end
    
        if points > 1
            subdata = data[1]
            MonotonicData = subdata[:,1]
            MonotonicDates = subdata[:,2]
            LabelPoint = subdata[1,3]
            Plots.scatter(MonotonicDates,MonotonicData .+ abs(minimum(MonotonicData)),xlabel="Date",ylabel="Monotonic Cumulative Displacment", label="Point Index: $(LabelPoint)")
    
            for i in 2:points
                subdata = data[i]
                MonotonicData = subdata[:,1]
                MonotonicDates = subdata[:,2]
                LabelPoint = subdata[1,3]
                outplot = Plots.scatter!(MonotonicDates,MonotonicData .+ abs(minimum(MonotonicData)),xlabel="Date",ylabel="Monotonic Cumulative Displacment", label="Point Index: $(LabelPoint)")
            end
        end
    elseif typeof(data) == Matrix{Any}
        MonotonicData = data[:,1]
        MonotonicDates = data[:,2]
        LabelPoint = data[1,3]
        outplot = Plots.scatter(MonotonicDates,MonotonicData .+ abs(minimum(MonotonicData)),xlabel="Date",ylabel="Monotonic Cumulative Displacment", label="Point Index: $(LabelPoint)")
    end
    return outplot
end

function MapMagnitudes(path::String, data::Vector{Any}, points; crop_size::Union{Tuple,Nothing}=nothing)
    color_image = TrueColor(path, crop_size=crop_size)[1]
    color_data = []

    # Extract x and y coordinates of edges
    x_edges = [x for (x, _) in points]
    y_edges = [y for (_, y) in points]

    for i in 1:length(data)
        subdata = data[i]
        disp = subdata[:,1] .+ abs(minimum(subdata[:,1]))
        mean_displacement = mean(disp)
        push!(color_data, mean_displacement)
    end
    
    mapped = Plots.scatter!(y_edges, x_edges, marker_z=color_data, label="Chip Centers")
    return mapped
end

function AverageSlumpSeries(data, name::String; labels::Bool=true, dataframe::Bool=false)
    flattened_data = vcat(data...)

    df = DataFrame(flattened_data, [:Value, :Date, :ID])

    df[:, :DayOfYear] = round.(dayofyear.(df.Date), digits=-1)

    mean_values = combine(groupby(df, :DayOfYear), :Value => mean => :mean_value, :Value => std => :std_value)

    plot(mean_values.DayOfYear,mean_values.mean_value, seriestype=:scatter, yerror=mean_values.std_value, 
        label="$(name)", color=:cornflowerblue, xticks=range(minimum(mean_values.DayOfYear), stop=maximum(mean_values.DayOfYear), step=10))
    p = plot!(mean_values.DayOfYear,mean_values.mean_value, linestyle=:dash, yerror=mean_values.std_value, 
        label="", color=:cornflowerblue)

    if labels
        xlabel!("Rounded Day of Year")
        ylabel!("Mean Displacement")
    end

    if dataframe
        return p,df
    else
        return p
    end
end