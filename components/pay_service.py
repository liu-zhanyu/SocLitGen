
import hashlib
import xmltodict
from motor.motor_asyncio import AsyncIOMotorClient
from client import *
from datetime import datetime

async def create_order_record(order_id: str, user_id: str, amount: int):
    client = AsyncIOMotorClient(
        f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
    )
    db = client.get_database("Newbit")
    orders_collection = db["orders"]
    await orders_collection.update_one(
        {"order_id": order_id},
        {
            "$setOnInsert": {
                "order_id": order_id,
                "user_id": user_id,
                "amount": amount,
                "status": "PENDING",
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )


async def mark_order_paid(order_id: str, transaction_id: str):
    client = AsyncIOMotorClient(
        f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
    )
    db = client.get_database("Newbit")
    orders_collection = db["orders"]
    await orders_collection.update_one(
        {"order_id": order_id},
        {
            "$set": {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "paid_at": datetime.utcnow()
            }
        }
    )


async def get_order_status(order_id: str):
    client = AsyncIOMotorClient(
        f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
    )
    db = client.get_database("Newbit")
    orders_collection = db["orders"]
    order = await orders_collection.find_one({"order_id": order_id})
    if not order:
        return {"status": "NOT_FOUND"}
    return {
        "status": order.get("status", "PENDING"),
        "order_id": order.get("order_id"),
        "amount": order.get("amount"),
        "transaction_id": order.get("transaction_id", None),
        "paid_at": order.get("paid_at", None)
    }


def generate_sign(params: dict, api_key: str) -> str:
    sorted_params = sorted((k, v) for k, v in params.items() if v)
    stringA = "&".join(f"{k}={v}" for k, v in sorted_params)
    stringSignTemp = f"{stringA}&key={api_key}"
    return hashlib.md5(stringSignTemp.encode("utf-8")).hexdigest().upper()


def dict_to_xml(params: dict) -> str:
    xml = ["<xml>"]
    for k, v in params.items():
        xml.append(f"<{k}><![CDATA[{v}]]></{k}>")
    xml.append("</xml>")
    return "".join(xml)


def xml_to_dict(xml_str: str) -> dict:
    return xmltodict.parse(xml_str).get("xml", {})
