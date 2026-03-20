from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "shopguard"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

# Collections
users_collection = db["users"]
alerts_collection = db["alerts"]


async def connect_db():
    """Create indexes on startup."""
    await users_collection.create_index("email", unique=True)
    await alerts_collection.create_index("timestamp")


def close_db():
    """Close MongoDB connection."""
    client.close()
