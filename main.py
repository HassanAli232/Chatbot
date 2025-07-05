from geojosn_reader import GeoRoadReader

# === Load both files ===
# path1 = "data/Jan 2022/Abu-Baker Al-Siddiq Rd NB_9.geojson"
# path2 = "data/Jan 2023/Abu-Baker Al-Siddiq Rd NB_9.geojson"

# gdf1 = read_geojson(path1)
# gdf2 = read_geojson(path2)

Reader = GeoRoadReader("data")

# all_files = Reader.read_all_geojson_files()
# roads = Reader.extract_road_metadata(all_files)

# print(Reader.get_geojson_files())
print(Reader.get_roads_metadata())
