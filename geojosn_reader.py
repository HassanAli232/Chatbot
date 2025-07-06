import geopandas as gpd
from pathlib import Path

class GeoRoadReader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self._geojson_files_paths = set()
        self.road_metadata = []

        # Load all GeoJSON files and extract metadata on initialization
        self._read_geojson_files_paths()
        self._extract_road_metadata()

    def _read_geojson_files_paths(self):
        """Recursively find all GeoJSON files under year directories."""
        self._geojson_files_paths = set(Path(self.data_dir).rglob("*.geojson"))

    def _extract_road_metadata(self):
        """
        Extract metadata from full paths:
        - 'road': cleaned name of the road (from filename)
        - 'year': from directory name like 'Jan 2022'
        - 'path': full path to the file
        """
        roads = []
        for path in self._geojson_files_paths:
            
            parts = Path(path).parts
            
            if len(parts) >= 3:
                year_dir = parts[-2]  # e.g., 'Jan 2022'

                filename = Path(path).stem  # e.g., 'Abu-Baker Al-Siddiq Rd NB_9'

                road_name = filename.split("_")[0]  # Clean up the road name
                
                roads.append({
                    "road": road_name.strip(),
                    "year": year_dir,
                    "path": str(path)
                })
        
        self.road_metadata = roads


    def read_geojson(self, path):
        """Reads and cleans a GeoJSON file into a GeoDataFrame."""
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notnull()].copy()
        return gdf

    def checkDifferences(self, gdf1, gdf2):
        """Checks for differences in two GeoDataFrames."""
        
        # === Align columns (excluding geometry) ===
        common_cols = sorted(list(set(gdf1.columns) & set(gdf2.columns) - {'geometry'}))
        df1 = gdf1[common_cols].reset_index(drop=True)
        df2 = gdf2[common_cols].reset_index(drop=True)

        # === Match row count for comparison ===
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len]
        df2 = df2.iloc[:min_len]

        # === Compare differences ===
        diff = df1.compare(df2)

        if diff.empty:
            print("✅ No content differences found.")
        else:
            print(f"🔍 Content differences found in {len(diff) // 2} row(s):\n")

            # Restructure for printing
            grouped = diff.stack(future_stack=True).reset_index()
            # Columns: ['row', 'column', 'version', 'value']
            for row in grouped['level_0'].unique():
                row_diff = grouped[grouped['level_0'] == row]
                print(f"--- Row {row} ---")
                for _, r in row_diff.iterrows():
                    print(_, r)
                print()

        # === Geometry comparison ===
        geom_diff_indices = [
            i for i in range(min_len)
            if not gdf1.geometry.iloc[i].equals(gdf2.geometry.iloc[i])
        ]

        if geom_diff_indices:
            print(f"\n🗺️ Geometry differences at rows: {geom_diff_indices}")
        else:
            print("\n✅ No geometry differences found.")


    def get_geojson_files_paths(self):
        """Return all GeoJSON file paths."""
        return self._geojson_files_paths.copy()
    
    def get_roads_metadata(self):
        """Return all road metadata extracted from GeoJSON files."""
        return self.road_metadata.copy()

    def find_versions_for_road(self, road_query):
        """Return all GeoJSON file entries that match the road name query."""
        return [
            r for r in self.road_metadata
            if road_query.lower() in r["road"].lower()
        ]
    
    
    def update_metadata(self):
        """Scan data directory and add metadata for new GeoJSON files not yet in metadata."""
        current_paths = {r['path'] for r in self.road_metadata}
        updated_files = list(Path(self.data_dir).rglob("*.geojson"))
        
        # Filter out files that are already in current metadata
        new_metadata = []
        for path in updated_files:
            if str(path) not in current_paths:
                parts = Path(path).parts
                
                if len(parts) >= 3:
                    year_dir = parts[-2]

                    filename = Path(path).stem

                    road_name = filename.split("_")[0]  # Clean up the road name

                    new_metadata.append({
                        "road": road_name.strip(),
                        "year": year_dir,
                        "path": str(path)
                    })

        self.road_metadata.extend(new_metadata)
        self._geojson_files_paths = list({*self._geojson_files_paths, *updated_files})
