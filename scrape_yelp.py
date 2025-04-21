from yelpapi import YelpAPI
import pandas as pd
import time

# STEP 1: Your Yelp API Key
API_KEY = "YOUR_YELP_API_KEY"  # Replace with your key
API_KEY = "i_NdUM3E9W65y17y4eStAa9aIApKxC79vT3IR4f2njme7J2uIj5skUhEadwblzrFdmEXCGL9OtsoDc-nvFV21nlJ9tDpEXdbqE3n3AML_-iFCgYs4YZ7vGjC0ur_Z3Yx"

yelp_api = YelpAPI(API_KEY)
location = "Fremont, CA"
limit = 50

# Step 1: Initial query to check total
initial = yelp_api.search_query(term="restaurants", location=location, limit=1)
total_available = min(initial["total"], 1000)  # Yelp only lets you access up to 1000
print(f"🔎 Total available results: {total_available}")

# Step 2: Loop over pages
all_data = []
for offset in range(0, total_available, limit):
    print(f"Fetching results {offset + 1} to {min(offset + limit, total_available)}...")

    try:
        response = yelp_api.search_query(
            term="restaurants",
            location=location,
            limit=limit,
            offset=offset,
            sort_by="rating",
        )
    except Exception as e:
        print(f"❌ Error at offset {offset}: {e}")
        break

    businesses = response.get("businesses", [])
    if not businesses:
        break

    for biz in businesses:
        all_data.append(
            {
                "name": biz["name"],
                "address": biz["location"]["address1"],
                "city": biz["location"]["city"],
                "state": biz["location"]["state"],
                "zip_code": biz["location"]["zip_code"],
                "latitude": biz["coordinates"]["latitude"],
                "longitude": biz["coordinates"]["longitude"],
                "rating": biz["rating"],
                "review_count": biz["review_count"],
                "categories": ", ".join([cat["title"] for cat in biz["categories"]]),
                "price": biz.get("price", "N/A"),
                "phone": biz.get("display_phone", "N/A"),
                "url": biz["url"],
            }
        )

    time.sleep(1)  # to avoid hitting rate limits

# Step 3: Save
df = pd.DataFrame(all_data)
df.to_csv("fremont_restaurants.csv", index=False)
print(f"Saved {len(df)} restaurants to fremont_restaurants.csv")
