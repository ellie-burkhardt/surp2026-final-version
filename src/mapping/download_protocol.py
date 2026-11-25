# For arcGISDownload()
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# For usgsDEMDownload()
from pathlib import Path
import zipfile
import rasterio
from rasterio.merge import merge
import shutil

# The topographic coordinates for Mount Hood should be 
# xmin = -121.79680
# ymin = 45.302316
# xmax = -121.597781
# ymax = 45.42716

def userInput():
    # Get coordinates from the user
    xmin = float(input("Enter west longitude (xmin): "))
    ymin = float(input("Enter south latitude (ymin): "))
    xmax = float(input("Enter east longitude (xmax): "))
    ymax = float(input("Enter north latitude (ymax): "))

    return xmin, ymin, xmax, ymax

def arcGISDownload(xmin, ymin, xmax, ymax): 
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        "World_Topo_Map/MapServer/export"
    )

    # Adjust aspect ratio
    lon_span = xmax - xmin
    lat_span = ymax - ymin

    aspect_ratio = lon_span / lat_span

    height = 1024
    width = int(height * aspect_ratio)

    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        # 4326 is the EPSG code for the WGS84 geographic coordinate system, the
        # standard latitude/ longitude system used by GPS
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "png",
        "transparent": "false",
        "f": "image"
    }

    # Generates a map covering the bounding box and sends back the image data
    response = requests.get(url, params = params)
    # Opens the downloaded image and loads it into a PIL Image object
    img = Image.open(BytesIO(response.content))
    # Converts the image to a NumPY array
    arr = np.array(img)

def usgsDEMDownload(demData): 
    print("\nPlease eunsure that the file you would like to use in in the 'downloads' folder on you device.")
    dataTXT = input("What is the name of the original .txt file? ")
   
    # Initial Settings
    BASE_DIR = Path.cwd()

    DOWNLOAD_DIR = BASE_DIR / "downloads"
    DOWNLOAD_DIR.mkdir(exist_ok = True)

    URL_FILE = DOWNLOAD_DIR / dataTXT
    OUTPUT_DEM = BASE_DIR / demData
    
    '''
    print(f"Working directory : {BASE_DIR}")
    print(f"URL file          : {URL_FILE}")
    print(f"Download folder   : {DOWNLOAD_DIR}")
    print(f"Output DEM        : {OUTPUT_DEM}")
    '''

    # Verify URL list exists
    if not URL_FILE.exists():
        raise FileNotFoundError(
            f"\nCould not find URL list:\n{URL_FILE}\n"
            f"\nPlace '{dataTXT}' inside:\n{DOWNLOAD_DIR}"
        )

    # Download files
    with open(URL_FILE, "r") as f:
        urls = [
            line.strip()
            for line in f
            if line.strip()
        ]
    
    print(f"Found {len(urls)} URLs")

    for url in urls:
        filename = url.split("/")[-1]
        filepath = DOWNLOAD_DIR / filename

        if filepath.exists():
            #print(f"Skipping existing file: {filename}")
            continue

        print(f"Downloading: {filename}")

        try: 
            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(filepath, "wb") as out:
                for chunk in r.iter_content(chunk_size = 1024*1024):
                    out.write(chunk)

        except Exception as e:
            print(
                f"Failed to download {filename}"
            )
            print(e)
    
    # Extract ZIP Files
    for zip_file in DOWNLOAD_DIR.glob("*.zip"):
        extract_dir = DOWNLOAD_DIR / zip_file.stem

        if extract_dir.exists():
            continue

        print(f"Extracting: {zip_file.name}")

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(extract_dir)

    # Locate all GeoTIFFs
    tif_files = list(DOWNLOAD_DIR.rglob("*.tif"))

    if not tif_files:
        raise RuntimeError("No GeoTIFF files found in the downloads folder")

    #print(f"Found {len(tif_files)} GeoTIFF files")

    for f in tif_files:
        try:
            with rasterio.open(str(f)) as src:
                src.read()
            #print(f"OK: {f}")
        except Exception as e:
            #print(f"BAD: {f}")
            print(e)

    # Validate DEM files
    valid_tifs = []
    bad_tifs = []

    for tif in tif_files:
        try:
            with rasterio.open(str(tif)) as src:
                '''
                print(f"Validating: {tif.name}")
                print(
                    f"  OK ({src.width} x {src.height}, "
                    f"{src.count} band(s))"
                )
                '''

            # Store the path, not the open dataset
            valid_tifs.append(tif)

        except Exception as e:
            bad_tifs.append(tif)

            try:
                tif.unlink()
                #print(f"Deleted corrupt file: {tif.name}")
            except Exception as delete_error:
                print(delete_error)
            try: 
                src.close()
            except: 
                pass
    
    # Report bad files
    if bad_tifs: 
        print("\n the following DEMS are unusuable:")

        for tls in bad_tifs: 
            print(f"    - {tls}")
        
        print(
            "\nPlease delete and re-download these files."
        )

    # Merge DEMS or Copy Single Tile
    if len(valid_tifs) == 1:
        #print("Only one DEM tile found. Copying to output...")
        shutil.copy2(valid_tifs[0], OUTPUT_DEM)
    else:
        demMerge(valid_tifs, OUTPUT_DEM)

    print(f"Done! Output DEM: {OUTPUT_DEM}")
    return OUTPUT_DEM

def demMerge(valid_tifs, OUTPUT_DEM):

    print(f"\nMerging {len(valid_tifs)} valid DEM tiles...")

    srcs = [rasterio.open(str(tif)) for tif in valid_tifs]

    try:
        mosaic, transform = merge(srcs)

        print("Mosaic shape:", mosaic.shape)

        for i, src in enumerate(srcs):
            print(i, src.bounds)

        meta = srcs[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "lzw"
        })

        with rasterio.open(OUTPUT_DEM, "w", **meta) as dest:
            dest.write(mosaic)
        
        from rasterio.transform import array_bounds

        bounds = array_bounds(
            mosaic.shape[1],
            mosaic.shape[2],
            transform
        )

        print("Merged bounds:", bounds)

    finally:
        for src in srcs:
            src.close()

def download():

    if hasattr(download, "_result"):
        return download._result
    
    location = input("Are you requesting avalanche information for Mount Hood? (yes/no): ").strip().lower()

    if location in {"yes", "y"}:
        xmin, ymin, xmax, ymax = (-121.79680, 45.302316, -121.597781, 45.42716)
        filename = f"MountHoodGEOTiff.tiff"
    else: 
        location = input("\nEnter the name of the mountain or forecast area you would like to analyze: ")

        print("\nThe area you selected is not covered by our automatic terrain mapping.")
        print("Mapping this area requires additional processing and may produce inaccurate results.")
        act_map = input("Would you like to skip terrain mapping and run the avalanche forecast only? (yes/no): ").strip().lower()

        if act_map == "yes": 
            return (location, None, None, None, None, None)

        print("Enter the minimum and maximum latitude and longitude coordinates for the area you would like to analyze.")
        xmin, ymin, xmax, ymax = userInput()
        filename = f"{location}GEOTiff.tiff"
    
    arcGISDownload(xmin, ymin, xmax, ymax)
    
    elevData = usgsDEMDownload(filename)

    download._result = (location, elevData, xmin, ymin, xmax, ymax)
    return download._result