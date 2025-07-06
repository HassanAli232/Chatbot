from geojosn_reader import GeoRoadReader


def get_roads_context(roads_names : list, reader: GeoRoadReader, versions: bool = False):

    """
    return a string with the context of the roads
    """
    road_contexts = {}

    for road in roads_names:
        
        context = get_road_context(road, reader, versions=versions)
        if context:
            road_contexts.update(context)
            
    return f"\n\nThe user might be referring to one of the roads \"{roads_names}\". Here is the known data:\n\n{road_contexts}"


def get_road_context(road_name, reader: GeoRoadReader, versions: bool = False):
    import json

    matches = reader.find_versions_for_road(road_name)
    
    if not matches:
        return None

    if not versions:
        # If versions are not requested, return the latest version only
        matches = [sorted(matches, key=lambda m: m["year"], reverse=True)[0]]
    else:
        # If versions are requested, return all versions
        matches = sorted(matches, key=lambda m: m["year"], reverse=True)
    
    summaries = {}
    
    
    for match in matches:
        path = match["path"]
        year = match["year"]


        gdf = reader.read_geojson(path)

        # if empty GeoDataFrame, skip to next match.
        if gdf.empty:
            continue

        total_distance = 0
        total_samples = 0
        weighted_speed_limit = 0
        weighted_avg_speed = 0
        weighted_median_speed = 0
        weighted_harmonic_speed = 0
        weighted_avg_time = 0

        # Compute weighted averages.
        for _, row in gdf.iterrows():
            
            results = row.get("segmentTimeResults")
            
            if isinstance(results, str):
                results = json.loads(results)
            
            if not isinstance(results, list) or len(results) == 0:
                continue

            rep = next((r for r in results if r.get("timeSet") == 4), results[0])
            samples = rep.get("sampleSize", 0)

            if not isinstance(samples, (int, float)) or samples <= 0:
                continue

            speed_limit = row.get("speedLimit", 0)
            weighted_speed_limit += (speed_limit or 0) * samples
            weighted_avg_speed += rep.get("averageSpeed", 0) * samples
            weighted_median_speed += rep.get("medianSpeed", 0) * samples
            weighted_harmonic_speed += rep.get("harmonicAverageSpeed", 0) * samples
            weighted_avg_time += rep.get("averageTravelTime", 0) * samples

            total_samples += samples
            total_distance += row.get("distance", 0)


        if total_samples != 0:
            avg_speed_limit = weighted_speed_limit / total_samples
            avg_speed = weighted_avg_speed / total_samples
            median_speed = weighted_median_speed / total_samples
            harmonic_speed = weighted_harmonic_speed / total_samples
            avg_time = weighted_avg_time / total_samples
            summary = f"""
📍 Road: {road_name} ({year})
- 📏 Total Distance: {total_distance/1000:.2f} km
- 🚘 Segments: {len(gdf)}
- 🧪 Total Samples: {total_samples}
- 🚦 Speed Limit: {avg_speed_limit:.1f} km/h
- 📊 Typical Avg Speed: {avg_speed:.1f} km/h
- 📈 Median Speed: {median_speed:.1f} km/h
- 🧮 Harmonic Speed: {harmonic_speed:.1f} km/h
- ⏱️ Avg Travel Time: {avg_time:.2f} sec
""".strip()
            
        else:
            summary = f"📍 Road: {road_name} ({year})\nNo valid data found."

        summaries[road_name + "," + str(year)] = summary

    return summaries
