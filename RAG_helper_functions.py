import difflib
import json

from geojosn_reader import GeoRoadReader

# === RAG Helpers ===
def get_best_road_match(user_input, available_roads):
    return difflib.get_close_matches(user_input, available_roads, n=1)



def get_road_context(road_name, reader: GeoRoadReader):
    import json

    matches = reader.find_versions_for_road(road_name)
    if not matches:
        return None

    # Get the latest year/version
    latest = sorted(matches, key=lambda m: m["year"], reverse=True)[0]
    gdf = reader.read_geojson(latest["path"])

    if gdf.empty:
        return None

    total_distance = 0
    total_samples = 0
    weighted_speed_limit = 0
    weighted_avg_speed = 0
    weighted_median_speed = 0
    weighted_harmonic_speed = 0
    weighted_avg_time = 0

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

        # Sum weighted values
        speed_limit = row.get("speedLimit", 0)
        weighted_speed_limit += (speed_limit or 0) * samples
        weighted_avg_speed += rep.get("averageSpeed", 0) * samples
        weighted_median_speed += rep.get("medianSpeed", 0) * samples
        weighted_harmonic_speed += rep.get("harmonicAverageSpeed", 0) * samples
        weighted_avg_time += rep.get("averageTravelTime", 0) * samples

        total_samples += samples
        total_distance += row.get("distance", 0)

    if total_samples == 0:
        return f"📍 Road: {road_name} ({latest['year']})\nNo valid data found."

    # Calculate averages
    avg_speed_limit = weighted_speed_limit / total_samples
    avg_speed = weighted_avg_speed / total_samples
    median_speed = weighted_median_speed / total_samples
    harmonic_speed = weighted_harmonic_speed / total_samples
    avg_time = weighted_avg_time / total_samples

    summary = f"""
📍 Road: {road_name} ({latest['year']})
- 📏 Total Distance: {total_distance/1000:.2f} km
- 🚘 Segments: {len(gdf)}
- 🧪 Total Samples: {total_samples}
- 🚦 Speed Limit: {avg_speed_limit:.1f} km/h
- 📊 Typical Avg Speed: {avg_speed:.1f} km/h
- 📈 Median Speed: {median_speed:.1f} km/h
- 🧮 Harmonic Speed: {harmonic_speed:.1f} km/h
- ⏱️ Avg Travel Time: {avg_time:.2f} sec
""".strip()

    return summary
