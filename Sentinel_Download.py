import ee
import datetime
import rasterio
import numpy as np
import matplotlib.pyplot as plt

ee.Initialize(project='earthengine-380314')

def sentinel_imagery(latitude, longitude, years, folder_path, start_month=5, end_month=10, buffer_size=500):
    """
    Export Sentinel-2 images with the lowest cloud cover for each week from May to October for specified years.
    
    Parameters:
    - latitude: Latitude of the point of interest.
    - longitude: Longitude of the point of interest.
    - years: List of years to process.
    - start_month: Starting month of the range (inclusive).
    - end_month: Ending month of the range (inclusive).
    - buffer_size: Buffer size in meters around the point.
    - folder_path: Google Drive folder path for exports.
    """
    # Create a point geometry
    point = ee.Geometry.Point([longitude, latitude])
    
    # Create a buffer around the point
    buffered_area = point.buffer(buffer_size)

    # Define a function to calculate the minimum cloud cover for each week
    def get_lowest_cloud_day(image_collection):
        def week_filter(start_date):
            end_date = start_date + datetime.timedelta(days=7)
            return ee.Filter.date(start_date.isoformat(), end_date.isoformat())

        weeks = []
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        while start < end:
            weeks.append(week_filter(start))
            start += datetime.timedelta(days=7)

        def select_min_cloud(week_filter):
            weekly_images = image_collection.filter(week_filter)
            return ee.Image(weekly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
        
        return [select_min_cloud(f) for f in weeks]

    # Loop over each year
    for year in years:
        start_date = f'{year}-{start_month:02d}-01'
        end_date = f'{year}-{end_month:02d}-30'
        
        # Load Sentinel-2 image collection and filter by date, location, and cloud cover
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(buffered_area) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

        # image.date().format("YYYY-MM-dd")

        # Cast the image to UInt16 to ensure consistent data types and select RGB and NIR bands
        def cast_to_uint16(image):
            return image.select(['B4', 'B3', 'B2', 'B8', 'B11']).toUint16()

        # Get the images with the lowest cloud cover for each week
        weekly_images = [cast_to_uint16(img) for img in get_lowest_cloud_day(sentinel2) if img]

        # Export each image to Google Drive as a GeoTIFF
        for image in weekly_images:
            # Get the date from the image's metadata
            try:
                image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                print("Exporting image with date:", image_date)

                task = ee.batch.Export.image.toDrive(
                    image=image,
                    description=f'Sentinel2_{image_date}',
                    scale=10,  # Sentinel-2 images have a resolution of 10 meters
                    region=buffered_area.getInfo()['coordinates'],
                    fileFormat='GeoTIFF',
                    folder=folder_path
                )
                task.start()
            except Exception as e:
                print("Error retrieving image date. Likely no images below the cloud threshold.", e)

    print(f"Export tasks have been started for years {years}. Check Google Drive for results.")

def display_tiff(file_path, bands=(1, 2, 3), stretch_percent=2):
    """
    Display a TIFF file using normalized RGB values and print metadata.

    Parameters:
    - file_path: Path to the TIFF file.
    - bands: A tuple indicating which bands to use for RGB display (default is (1, 2, 3) for R, G, B).
    - stretch_percent: Percent of pixel values to stretch for contrast (default is 2%).
    """
    with rasterio.open(file_path) as src:
        # Read the specified bands
        red = src.read(bands[0])
        green = src.read(bands[1])
        blue = src.read(bands[2])
        
        # Stack the bands into a single array
        rgb = np.stack((red, green, blue), axis=-1)

        # Normalize the bands
        rgb = normalize(rgb, stretch_percent)

        # Extract metadata
        metadata = src.meta
        transform = src.transform
        pixel_size_x = transform[0]
        pixel_size_y = -transform[4]  # Usually negative for geographic north-up images

        # Print metadata information
        print(f"File: {file_path}")
        print(f"Width: {metadata['width']} pixels")
        print(f"Height: {metadata['height']} pixels")
        print(f"Number of bands: {metadata['count']}")
        print(f"Driver: {metadata['driver']}")
        print(f"Coordinate Reference System: {metadata['crs']}")
        print(f"Pixel Size: {pixel_size_x} x {pixel_size_y} meters per pixel")

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title('Normalized RGB Image')
        plt.axis('off')
        plt.show()
        
def normalize(image, stretch_percent):
    """
    Normalize an image array for display with optional stretching.

    Parameters:
    - image: The image array to normalize.
    - stretch_percent: Percent of pixel values to stretch for contrast (default is 2%).

    Returns:
    - Normalized image.
    """
    # Compute the stretch limits
    lower_percentile = np.percentile(image, stretch_percent)
    upper_percentile = np.percentile(image, 100 - stretch_percent)
    
    # Stretch the image to the 0-1 range
    image = np.clip((image - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)

    return image
