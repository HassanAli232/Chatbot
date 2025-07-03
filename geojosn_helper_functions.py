import geopandas as gpd

def read_geojson(path):
    """Reads and cleans a GeoJSON file into a GeoDataFrame."""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    return gdf


def checkDifferences(gdf1, gdf2):
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

