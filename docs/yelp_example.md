# Yelp Open Dataset — Format Reference

> **Note:** The examples below are **synthetic** — fabricated to illustrate the
> dataset's structure and encoding quirks. They are not reproduced from the actual
> Yelp Open Dataset, which may not be redistributed under its
> [terms of use](https://business.yelp.com/data/resources/open-dataset/).

---

## business.json

One JSON object per line. Each object represents a single business.

```json
{
  "business_id": "vcNAWiLM4dR7D2nwwJ7nCA",
  "name":         "Bayou Jazz Kitchen",
  "city":         "New Orleans",
  "state":        "LA",
  "stars":        4.5,
  "review_count": 312,
  "is_open":      1,
  "categories":   "Restaurants, Cajun/Creole, Jazz & Blues",
  "latitude":     29.9511,
  "longitude":    -90.0715,

  "attributes": {
    "NoiseLevel":               "u'loud'",
    "RestaurantsGoodForGroups": "True",
    "RestaurantsReservations":  "True",
    "RestaurantsPriceRange2":   "2",
    "OutdoorSeating":           "True",
    "Alcohol":                  "u'full_bar'",
    "GoodForKids":              "False",
    "HappyHour":                "True",
    "HasTV":                    "False",
    "Caters":                   "True",
    "WheelchairAccessible":     "True",
    "DogsAllowed":              "False",
    "WiFi":                     "u'free'",
    "RestaurantsAttire":        "u'casual'",
    "BYOB":                     "False",
    "Corkage":                  "False",
    "Ambience":         "{'romantic': False, 'intimate': False, 'classy': True, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': True, 'upscale': False, 'casual': True}",
    "GoodForMeal":      "{'breakfast': False, 'brunch': True, 'lunch': True, 'dinner': True, 'latenight': False, 'dessert': False}",
    "Music":            "{'live': True, 'dj': False, 'jukebox': False, 'karaoke': False, 'background_music': False, 'no_music': False, 'video': False}",
    "BusinessParking":  "{'garage': False, 'street': True, 'lot': True, 'valet': False, 'validated': False}"
  },

  "hours": {
    "Tuesday":   "11:0-22:0",
    "Wednesday": "11:0-22:0",
    "Thursday":  "11:0-23:0",
    "Friday":    "11:0-0:0",
    "Saturday":  "10:0-0:0",
    "Sunday":    "10:0-21:0"
  }
}
```

### Attribute encoding quirks

All values inside `attributes` arrive as **strings**, regardless of their logical type.
The ingest pipeline (`ingestion/ingest_nola.py`) normalises three distinct encodings:

| Encoding | Example raw value | Parsed as | How |
|---|---|---|---|
| Plain string boolean | `"True"` / `"False"` | `1` / `0` (SQLite INTEGER) | direct cast |
| Python 2 unicode repr | `"u'loud'"` / `"u'full_bar'"` | `"loud"` / `"full_bar"` (TEXT) | strip `u'...'` wrapper via regex |
| Python repr dict | `"{'live': True, 'dj': False, ...}"` | `{"live": true, "dj": false, ...}` (JSON TEXT) | `ast.literal_eval()` → `json.dumps()` |

The dict-encoded fields (`Ambience`, `GoodForMeal`, `Music`, `BusinessParking`) are
stored as JSON strings in SQLite and queried with `json_extract()`:

```sql
-- restaurants with live music
SELECT name FROM businesses WHERE json_extract(music, '$.live') = 1;

-- jazz brunch spots
SELECT name FROM businesses
WHERE json_extract(good_for_meal, '$.brunch') = 1
  AND json_extract(music, '$.live') = 1;
```

---

## review.json

One JSON object per line. Each object is a single review, linked to a business via `business_id`.

```json
{"review_id": "r_001", "business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "stars": 5, "date": "2024-03-15", "text": "The jazz trio started at 9pm and the whole room came alive. Loud in the best possible way — we could still hear each other over the trumpet. Gumbo was the best I've had in New Orleans."}
{"review_id": "r_002", "business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "stars": 4, "date": "2024-01-20", "text": "Great for a big group. We had 14 people for a bachelorette and they handled it without breaking a sweat. Happy hour cocktails are generous."}
{"review_id": "r_003", "business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "stars": 5, "date": "2023-11-08", "text": "Sunday jazz brunch is a must. Live band, bottomless mimosas, outdoor seating under the oak tree. This is what New Orleans is about."}
{"review_id": "r_004", "business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "stars": 3, "date": "2023-09-14", "text": "Food is solid, service was slow for our large table. The music makes up for it. Very loud — not ideal if you want a quiet conversation."}
{"review_id": "r_005", "business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "stars": 5, "date": "2024-02-28", "text": "Came for the crawfish étouffée, stayed for the second line. The trumpet player walked between tables during the late set. Unforgettable."}
```

Reviews link to businesses via `business_id`. The pipeline embeds the `text` field
and stores it in ChromaDB with `business_id`, `stars`, and `date` as metadata —
enabling filtered vector search within a SQL-filtered candidate pool.

---

## How a query uses both files

```
"jazz brunch spot for a bachelorette party"
          │
          ▼
  Query Planner extracts:
    sql_filters  → good_for_meal.brunch=true, good_for_groups=true
    semantic_query → "bachelorette celebration festive lively atmosphere"
          │
          ├─ SQL filter on businesses table
          │    json_extract(good_for_meal, '$.brunch') = 1
          │    AND good_for_groups = 1
          │    AND stars >= 4.0
          │    → candidate pool (e.g. 6 businesses)
          │
          └─ ChromaDB vector search on review embeddings
               where business_id IN (candidate pool)
               → top 20 review snippets ranked by semantic similarity
                 e.g. "Great for a big group... bachelorette..."
                      "Sunday jazz brunch is a must..."
```
