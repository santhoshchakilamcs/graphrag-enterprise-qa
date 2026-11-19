"""Load the Northwind CSVs and expose lookup indexes.

All tables are plain CSVs under data/. Nothing here needs a database
server or an API key.
"""

from __future__ import annotations

import os
import re

import pandas as pd

TABLES = [
    "categories",
    "customers",
    "employees",
    "order_details",
    "orders",
    "products",
    "shippers",
    "suppliers",
]

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_data(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Read every Northwind CSV into a dict of DataFrames."""
    data_dir = data_dir or os.path.abspath(DEFAULT_DATA_DIR)
    tables = {}
    for name in TABLES:
        path = os.path.join(data_dir, f"{name}.csv")
        tables[name] = pd.read_csv(path)
    # Normalise the order dates so year filtering works.
    tables["orders"]["orderDate"] = pd.to_datetime(tables["orders"]["orderDate"])
    tables["orders"]["year"] = tables["orders"]["orderDate"].dt.year
    return tables


def full_name(first: str, last: str) -> str:
    return f"{first} {last}"


class Indexes:
    """Name -> id lookups for every entity type, used by the QA parser."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables
        self.categories = {
            str(row.categoryName): int(row.categoryID)
            for row in tables["categories"].itertuples()
        }
        self.suppliers = {
            str(row.companyName): int(row.supplierID)
            for row in tables["suppliers"].itertuples()
        }
        self.customers = {
            str(row.companyName): str(row.customerID)
            for row in tables["customers"].itertuples()
        }
        self.products = {
            str(row.productName): int(row.productID)
            for row in tables["products"].itertuples()
        }
        self.employees = {
            full_name(str(row.firstName), str(row.lastName)): int(row.employeeID)
            for row in tables["employees"].itertuples()
        }
        self.employee_last = {
            str(row.lastName): int(row.employeeID)
            for row in tables["employees"].itertuples()
        }
        self.shippers = {
            str(row.companyName): int(row.shipperID)
            for row in tables["shippers"].itertuples()
        }
        # Longest names first so "United Package" does not shadow a
        # longer match, and so "Margaret Peacock" beats "Peacock".
        self.all_names = sorted(
            list(self.categories)
            + list(self.suppliers)
            + list(self.customers)
            + list(self.products)
            + list(self.employees)
            + list(self.shippers),
            key=len,
            reverse=True,
        )

    def find_names(self, text: str) -> list[tuple[str, str]]:
        """Return (entity_type, name) for every known name in the text.

        Matching is case-insensitive substring matching on word
        boundaries. Longer names are tried first.
        """
        lowered = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
        found: list[tuple[str, str]] = []
        used_spans: list[tuple[int, int]] = []
        for name in self.all_names:
            needle = " " + re.sub(r"[^a-z0-9 ]", " ", name.lower()) + " "
            start = lowered.find(needle)
            if start == -1:
                continue
            end = start + len(needle)
            if any(start < u_end and end > u_start for u_start, u_end in used_spans):
                continue
            used_spans.append((start, end))
            if name in self.categories:
                found.append(("category", name))
            elif name in self.suppliers:
                found.append(("supplier", name))
            elif name in self.customers:
                found.append(("customer", name))
            elif name in self.products:
                found.append(("product", name))
            elif name in self.employees:
                found.append(("employee", name))
            elif name in self.shippers:
                found.append(("shipper", name))
        return found

    def find_years(self, text: str) -> list[int]:
        return [int(y) for y in re.findall(r"\b(199[6-8])\b", text)]
