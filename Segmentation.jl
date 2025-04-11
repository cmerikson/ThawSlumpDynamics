using Images, ImageView, ImageSegmentation, ImageDraw, ImageMorphology, FileIO, Plots, PyCall, Glob, DataFrames, Dates, Statistics

# Function for calling band selection from python script
function select_bands(file::String, bandA::Int, bandB::Int, bandC::Union{Nothing, Int}, output::String)
    py_file_path = joinpath(@__DIR__, "Select_Bands.py")
    @pyinclude(py_file_path)
    py"select_bands"(file,bandA,bandB,bandC,output)
end

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
function preview(file::String; crop_size::Union{Nothing, Tuple{Int, Int}}=nothing)
    img = load(file)
    if crop_size != nothing
        img = crop_center(img, crop_size)
    end
    imshow(img)
end

# Function to draw line on image
function draw_line(img::AbstractArray, modifications::Vector{Tuple{Real, Real, Vararg{Float64}}})
    if eltype(img) <: Gray
        img = colorview(RGB, img, img, img)
    end
    
    line = LineSegment(CartesianIndex(modifications[1][1],modifications[1][2]), CartesianIndex(modifications[2][1],modifications[2][2]))
    color = RGB{N0f8}(modifications[3][1], modifications[3][2], modifications[3][3])
    temp = copy(img)
    temp = draw!(temp, line, color)
    return temp
end

# Function to display segments with a title
function display_segments(segments, file_path::String)
    segment = map(i -> segment_mean(segments, i), labels_map(segments))
    file_name = splitext(basename(file_path))[1]
    title_text = last(file_name, 10)
    plot_title = "$title_text"
    plot = Plots.plot(segment, framestyle=:none, title=plot_title)
    display(plot)
end

# Function to get pixel count of segmented area
function count_pixels(object::Union{String, Matrix{RGB{N0f8}}}, Seed1::Tuple{Int64,Int64}, Seed2::Tuple{Int64,Int64}; Seed3::Union{Nothing, Tuple{Int64,Int64}} = nothing, Seed4::Union{Nothing, Tuple{Int64,Int64}} = nothing, Display::Bool = false, crop_size::Union{Nothing, Tuple{Int, Int}}=nothing, mods::Union{Nothing, Vector{Tuple{Real, Real, Vararg{Float64}}}}=nothing, water_mask::Union{Nothing, BitMatrix}=nothing, ndvi_threshold::Union{Nothing, Float64}=nothing, verbose::Bool = false)

    # Load image from file path if the input is a string
    if typeof(object) == String
        file_path = object
        if endswith(file_path, ".png")
            img = load(file_path)
        else
            println("File is not a .png: $file_path")
            return nothing
        end
    elseif typeof(object) == Matrix{RGB{N0f8}}
        img = object
    else
        println("Error: The object is not a valid path or image.")
        return nothing
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
        display_segments(segments,file_path)
    end

    if verbose
        println("The segemented region contains $pixel_count pixels.")
    end

    return pixel_count, segments
end

#orig_colormap = cgrad(:linear_protanopic_deuteranopic_kbjyw_5_95_c25_n256)
#reversed_colormap = reverse(orig_colormap.colors)

# Create a new colormap where zero is black
#custom_colormap = cgrad(vcat(RGB(0, 0, 0), reversed_colormap[2:end]))

function segment_folder(folder::String, Seed1::Tuple{Int64,Int64}, Seed2::Tuple{Int64,Int64}; Seed3::Union{Nothing, Tuple{Int64,Int64}} = nothing, Seed4::Union{Nothing, Tuple{Int64,Int64}} = nothing, Display::Bool = false, ndvi::Bool=false, ndwi_threshold::Union{Nothing, Float64} = nothing, ndwi_image::Union{Nothing, String}=nothing, crop_size::Union{Nothing, Tuple{Int, Int}}=nothing, mods::Union{Nothing, Vector{Tuple{Real, Real, Vararg{Float64}}}}=nothing, verbose::Bool = false, dilate_water::Bool = false, dilation::Int=1, refine::Bool = false)
    
    # Get all files with tif extension
    files = glob("*.tif", folder)
    
    if isempty(files)
        println("No files found. Check folder path and file naming conventions.")
        return nothing
    end

    date_format = DateFormat("yyyy-mm-dd")
    
    # Function to extract and parse date from the last ten characters of filename
    function extract_date(filename)
        base = basename(filename) # Remove folder path
        date_str = base[11:20] # Get the last ten characters excluding extension
        return Date(date_str, date_format) # Parse date
    end

    # Sort files by the extracted date in reverse order
    sorted = sort(files, by = x -> extract_date(x), rev = true)

    file_count = length(sorted)

    # Initialize result storage
    results = DataFrame()
    mask = nothing
    outlines = heatmap(framestyle=:none)
    masks = 0
    img_size = size(load(first(sorted)))
    consistent_pixels_mask = ones(Float64,img_size)

     if crop_size != nothing
        consistent_pixels_mask = crop_center(consistent_pixels_mask, crop_size)
     end  

    # Create a temporary directory
    temp_dir = mktempdir()

    # NDWI Water Mask
    if ndwi_image != nothing
        selected_image = filter(s -> occursin(ndwi_image, s), files)
        if length(selected_image) == 1
            selected_image = first(selected_image) # Convert from :Vector{String} to ::String
            ndwi_path = select_bands(selected_image, 3,4,nothing, joinpath(temp_dir, "ndwi.png"))
            ndwi = load(ndwi_path)
            binary_ndwi = ndwi .> ndwi_threshold
            if dilate_water
                water_mask = .!dilate(binary_ndwi, r=dilation)
            else
                water_mask = .!binary_ndwi
            end
            if crop_size != nothing
                water_mask = crop_center(water_mask, crop_size)
            end
        elseif length(selected_image) > 1
            error("Error: There are multiple files with names matching the image selected for water masking.")
        elseif length(selected_image) == 0
            error("Error: Provided raster for water mask not found.")
        end
    else
        print("Defaulting to no water mask.")
        water_mask = 1.0
    end

    # Create Mask of Consistently Labeled Pixels
    if refine
        for i in [1:1:file_count;]
            rgb_path = select_bands(sorted[i], 1,2,3, joinpath(temp_dir, "rgb.png"))
    
            segments = count_pixels(rgb_path, Seed1, Seed2, Seed3=Seed3, Seed4=Seed4, crop_size=crop_size, mods=mods, verbose=verbose, Display=false)[2]
            slump_mask = labels_map(segments) .== 1
            consistent_pixels_mask = consistent_pixels_mask .* slump_mask
        end
        consistent_pixels_mask = dilate(consistent_pixels_mask, r=dilation)
        consistent_pixels_mask = consistent_pixels_mask .> 0.0  
    end

    # Main Segmentation
    for i in [1:1:file_count;]
        rgb_path = select_bands(sorted[i], 1,2,3, joinpath(temp_dir, "rgb.png"))
        ndvi_path = select_bands(sorted[i], 4,1,nothing, joinpath(temp_dir, "ndvi.png"))

        date = extract_date(sorted[i])

        if ndvi
            pixels_count, segments = count_pixels(ndvi_path, Seed1, Seed2, Seed3=Seed3, Seed4=Seed4, Display=false, crop_size=crop_size, mods=mods, verbose=verbose)
            
            row = DataFrame(Date=date, Pixels=pixel_count)
            append!(results, row)

            mask = labels_map(segments) .== 1
            mask = mask .* water_mask
    
            if Display
                masks = masks .+ mask
                outlines = heatmap!(reverse(masks, dims=1), color=cgrad(:linear_protanopic_deuteranopic_kbjyw_5_95_c25_n256, rev=true), colorbar=false)
            end

        else
            segments = count_pixels(rgb_path, Seed1, Seed2, Seed3=Seed3, Seed4=Seed4, crop_size=crop_size, Display=false, mods=mods, verbose=verbose)[2]
            mask = labels_map(segments) .== 1
            mask = mask .* water_mask

            if consistent_pixels_mask != nothing
                mask = mask .* consistent_pixels_mask
            end

            pixel_count = count(x -> x != 0, mask)
            row = DataFrame(Date=date, Pixels=pixel_count)
            append!(results, row)
            
            if Display
                masks = masks .+ mask
                outlines = heatmap!(reverse(masks, dims=1), color=cgrad(:linear_protanopic_deuteranopic_kbjyw_5_95_c25_n256, rev=true), colorbars=false)
            end
        end
    end

    # Clean up by deleting the temporary directory and its contents
    rm(temp_dir, recursive=true)

    if Display
        display(outlines)
        results = sort!(results, :Date)
        return results
    end
    
    #results.Date = Date.(results.Date, "yyyy-mm-dd")
    results = sort!(results, :Date)
    return results, outlines
end
