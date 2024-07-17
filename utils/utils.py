import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.windows import Window
import os
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import torch

class OutputGenerator:
    """
    A class to generate output shapefiles based on model predictions.

    Attributes:
        model: The trained PyTorch model used for making predictions.
        test_loader: DataLoader for the test dataset.
        dataset: Dataset containing class information.
        grid: GeoDataFrame containing grid data.
        result_path: Path to save the output shapefile.
        device: Device on which to run the model (CPU or GPU).
    """

    def __init__(self, model, test_loader, dataset, grid_file_path, result_path="outputs"):
        """
        Initializes the OutputGenerator with the given parameters.

        Args:
            model: The trained PyTorch model.
            test_loader: DataLoader for the test dataset.
            dataset: Dataset containing class information.
            grid_file_path: Path to the grid file.
            result_path: Path to save the output shapefile.
        """
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.grid = gpd.read_file(grid_file_path)
        self.result_path = result_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_shapefile(self):
        """
        Generates a shapefile with predicted classes for each grid cell.
        """
        predicted_f = []
        predicted_r = []

        with torch.no_grad():
            for images, labels, img_paths in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                for i in range(outputs.shape[0]):
                    output_i = outputs[i]
                    label_i = labels[i]

                    img_path_i = img_paths[i]
                    img_path_i = (img_path_i.split(".")[0]).split("/")[-1]
                    _, predicted = torch.max(output_i.data, 0)
                    
                    if predicted == self.dataset.classes.index("favelas"):
                        predicted_f.append(img_path_i)
                    if predicted == self.dataset.classes.index("residential"):
                        predicted_r.append(img_path_i)

        predicted_f = [int(i) for i in predicted_f]
        predicted_r = [int(i) for i in predicted_r]

        gdf = gpd.GeoDataFrame(geometry=self.grid.geometry)
        gdf["class"] = None
        gdf["id"] = self.grid["id"]

        for index, row in self.grid.iterrows():
            if row["id"] in predicted_f:
                gdf.loc[index, "class"] = "favelas"
            elif row["id"] in predicted_r:
                gdf.loc[index, "class"] = "residential"
            else:
                gdf.loc[index, "class"] = "others"

        gdf.to_file(self.result_path)
        
class Merger:
    """
    A class to merge multiple raster images into a single raster image.

    Attributes:
        image_paths: List of paths to the raster images to be merged.
        output_path: Path to save the merged raster image.
    """

    def __init__(self, image_paths, output_path=None):
        """
        Initializes the Merger with the given image paths and output path.

        Args:
            image_paths: List of paths to the raster images.
            output_path: Path to save the merged raster image. Defaults to a combination of input image names.
        """
        self.image_paths = image_paths

        if output_path is None:
            output_path = f"{image_paths[0].split('.')[0].split('/')[-1]}_{image_paths[1].split('.')[0].split('/')[-1]}.tif"
        self.output_path = output_path

    def merge_images(self):
        """
        Merges the raster images specified in image_paths and saves the result to output_path.
        """
        sources = [rasterio.open(path) for path in self.image_paths]

        merged, out_trans = merge(sources)

        merged_meta = sources[0].meta.copy()
        merged_meta.update({
            'transform': out_trans,
            'width': merged.shape[2],
            'height': merged.shape[1]
        })

        with rasterio.open(self.output_path, "w", **merged_meta) as dest:
            dest.write(merged)

    def show_merged_image(self):
        """
        Displays the merged raster image.
        """
        with rasterio.open(self.output_path) as merged_src:
            fig, ax = plt.subplots()
            show(merged_src, ax=ax, transform=merged_src.transform)
            ax.axis('off')
            plt.show()

class GridGenerator:
    """
    A class to generate a grid of polygons over a raster image.

    Attributes:
        image_path: Path to the raster image.
        taille_carreau: Size of each grid cell in the same units as the raster's CRS.
        output_grid_path: Path to save the generated grid shapefile.
    """

    def __init__(self, image_path, output_grid_path=None, taille_carreau=150):
        """
        Initializes the GridGenerator with the given parameters.

        Args:
            image_path: Path to the raster image.
            output_grid_path: Path to save the generated grid shapefile. Defaults to the same name as the image with a .shp extension.
            taille_carreau: Size of each grid cell in the same units as the raster's CRS.
        """
        self.image_path = image_path
        self.taille_carreau = taille_carreau
        if output_grid_path is None:
            output_grid_path = f"{image_path.split('.')[0]}.shp"
        self.output_grid_path = output_grid_path

    def generate_grid(self):
        """
        Generates a grid of polygons over the raster image and saves it to a shapefile.
        """
        with rasterio.open(self.image_path) as src:
            bounds = src.bounds

        # Define the size of the grid cell
        xmin, ymin, xmax, ymax = bounds
        rows = int((ymax - ymin) / self.taille_carreau)
        cols = int((xmax - xmin) / self.taille_carreau)
        polys = []

        for i in range(cols):
            for j in range(rows):
                # Calculate the geographic coordinates of the polygon corners
                x_left = xmin + i * self.taille_carreau
                x_right = xmin + (i + 1) * self.taille_carreau
                y_top = ymax - j * self.taille_carreau
                y_bottom = ymax - (j + 1) * self.taille_carreau

                # Create the polygon with the geographic coordinates
                poly = Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)])
                polys.append(poly)

        grid_gdf = gpd.GeoDataFrame(geometry=polys, crs=src.crs)

        grid_gdf.to_file(self.output_grid_path)

class Filter:
    """
    A class to filter a grid of polygons based on a raster image.

    Attributes:
        grid_shapefile_path: Path to the shapefile containing the grid.
        image_path: Path to the raster image.
        output_filtered_shapefile_path: Path to save the filtered grid shapefile.
    """

    def __init__(self, grid_shapefile_path, image_path, output_filtered_shapefile_path=None):
        """
        Initializes the Filter with the given parameters.

        Args:
            grid_shapefile_path: Path to the shapefile containing the grid.
            image_path: Path to the raster image.
            output_filtered_shapefile_path: Path to save the filtered grid shapefile. Defaults to the input grid name with a '_f' suffix.
        """
        self.grid_shapefile_path = grid_shapefile_path
        self.image_path = image_path
        if output_filtered_shapefile_path is None:
            output_filtered_shapefile_path = f"{grid_shapefile_path.split('.')[0]}_f.shp"
        self.output_filtered_shapefile_path = output_filtered_shapefile_path

    def filter_grid(self):
        """
        Filters the grid of polygons based on the presence of null pixels in the raster image
        and saves the filtered grid to a shapefile.
        """
        grid_gdf = gpd.read_file(self.grid_shapefile_path)

        with rasterio.open(self.image_path) as src:
            grid_gdf = grid_gdf.to_crs(src.crs)

            indices_to_remove = []

            for idx, poly in grid_gdf.iterrows():
                geom = poly.geometry
                window = src.window(*geom.bounds)
                subset = src.read(window=window)

                if (subset == 0).any():
                    indices_to_remove.append(idx)

        grid_gdf = grid_gdf.drop(indices_to_remove)

        grid_gdf.to_file(self.output_filtered_shapefile_path)

class Cutter:
    def __init__(self, vector_file, raster_file, output_folder):
        """
        Initializes the Cutter class with vector and raster files, and the output folder.

        Parameters:
        vector_file (str): Path to the vector file.
        raster_file (str): Path to the raster file.
        output_folder (str): Folder where the cut images will be saved.
        """
        self.vector_file = vector_file
        self.raster_file = raster_file
        self.output_folder = output_folder
        
    def class_carreau(self, row, p_fav):
        """
        Determines the class of a tile based on several criteria.

        Parameters:
        row (GeoSeries): A row from the GeoDataFrame containing tile information.
        p_fav (float): Threshold to classify a tile as favelas.

        Returns:
        str: The class of the tile (residential, favelas, vegetation, others).
        """
        if row["res"] == 1.0 and row["p_vegeta"] < 0.95 and row["ghsl"] > 0.5 and row["zi"] == 0.0:
            return "residential"
        if row["res"] == 0.0 and row["p_favelas"] >= p_fav and row["p_vegeta"] < 0.95 and row["ghsl"] > 0.5 and row["zi"] == 0.0:
            return "favelas"
        if row["p_vegeta"] >= 0.95:
            return "vegetation"
        return "others"
        
    def cut_images(self, p_fav):
        """
        Cuts the raster images based on the geometries from the vector file and classifies them.

        Parameters:
        p_fav (float): Threshold to classify a tile as favelas.
        """
        couche_vecteur = gpd.read_file(self.vector_file)

        gdf = gpd.GeoDataFrame(geometry=couche_vecteur.geometry)
        gdf["class"] = None
        gdf["id"] = couche_vecteur["id"]

        with rasterio.open(self.raster_file) as src:
            for index, row in couche_vecteur.iterrows():
                identifiant_carreau = row["id"]
                classe_carreau = self.class_carreau(row, p_fav)
                geom = row.geometry

                gdf.loc[index, "class"] = classe_carreau

                window = src.window(*geom.bounds)
                subset = src.read(window=window)

                profile = src.profile
                profile.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                destination_folder = os.path.join(self.output_folder, classe_carreau.lower())
                os.makedirs(destination_folder, exist_ok=True)

                cut_image_path = os.path.join(destination_folder, f"{identifiant_carreau}.tif")
                with rasterio.open(cut_image_path, "w", **profile) as dst:
                    dst.write(subset)
                    
        gdf.to_file(os.path.join(self.output_folder, "check"))
