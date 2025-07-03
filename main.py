from geojosn_helper_functions import read_geojson

# === Load both files ===
path1 = "data/Jan 2022/Abu-Baker Al-Siddiq Rd NB_9.geojson"
path2 = "data/Jan 2023/Abu-Baker Al-Siddiq Rd NB_9.geojson"

gdf1 = read_geojson(path1)
gdf2 = read_geojson(path2)

print(gdf1.head())
print("==" * 20)
print(gdf2.head())