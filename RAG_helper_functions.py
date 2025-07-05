from geojosn_reader import GeoRoadReader
import numpy as np

# Precompute embeddings for all available roads once (run once at startup)
def embed_texts(texts, client):
    embeddings = []
    batch_size = 10  # or suitable batch size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings


import numpy as np

def get_best_road_match(user_input, available_roads, available_roads_embeddings, client, top_k=3, threshold=0.6):
    # Embed the user input
    response = client.embeddings.create(
        input=[user_input],
        model="text-embedding-ada-002"
    )
    user_embedding = response.data[0].embedding

    # Compute cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_sim(user_embedding, emb) for emb in available_roads_embeddings]

    # Sort indices by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    # Collect top matches that meet the threshold
    matches = [
        available_roads[i]
        for i in sorted_indices[:top_k]
        if similarities[i] >= threshold
    ]

    return matches


def get_roads_context(roads_names, reader: GeoRoadReader):
    """
    Get context for all available roads.
    Returns a dictionary mapping road names to their context summaries.
    """
    road_contexts = {}
    for road in roads_names:
        context = get_road_context(road, reader)
        if context:
            road_contexts[road] = context
    
    road_contexts = f"\n\nThe user might be referring to one of the roads \"{roads_names}\". Here is the known data:\n\n{road_contexts}"

    
    return road_contexts

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
