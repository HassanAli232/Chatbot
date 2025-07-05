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


def get_best_road_match(user_input, available_roads, available_roads_embeddings, client):
    # Embed the user input
    response = client.embeddings.create(
        input=[user_input],
        model="text-embedding-ada-002"
    )
    user_embedding = response.data[0].embedding

    # Compute cosine similarity between user_embedding and all road embeddings
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_sim(user_embedding, emb) for emb in available_roads_embeddings]

    # Get index of best match
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    # You can set a threshold to avoid weak matches
    threshold = 0.7
    if best_score >= threshold:
        return [available_roads[best_idx]]
    else:
        return []



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
