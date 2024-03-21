import csv
import pathlib
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

DATA_DIR = pathlib.Path(__file__).absolute().parent
CUSTOMERS_FILE = DATA_DIR.joinpath("raw_customers.csv")
TRANSACTIONS_FILE = DATA_DIR.joinpath("raw_transactions.csv")

# https://www.babycenter.com/baby-names/most-popular/top-baby-names-2024
names = [
    "Olivia",
    "Amelia",
    "Sophia",
    "Emma",
    "Charlotte",
    "Isabella",
    "Ava",
    "Aurora",
    "Luna",
    "Mia",
    "Ellie",
    "Evelyn",
    "Lily",
    "Harper",
    "Nova",
    "Camila",
    "Mila",
    "Sofia",
    "Gianna",
    "Aria",
    "Scarlett",
    "Eliana",
    "Layla",
    "Violet",
    "Willow",
    "Ella",
    "Hazel",
    "Avery",
    "Nora",
    "Penelope",
    "Eleanor",
    "Elena",
    "Chloe",
    "Delilah",
    "Isla",
    "Ivy",
    "Abigail",
    "Elizabeth",
    "Riley",
    "Paisley",
    "Maya",
    "Zoey",
    "Aaliyah",
    "Serenity",
    "Kinsley",
    "Lucy",
    "Victoria",
    "Iris",
    "Lainey",
    "Grace",
    "Athena",
    "Ayla",
    "Naomi",
    "Leilani",
    "Emily",
    "Valentina",
    "Emilia",
    "Nevaeh",
    "Stella",
    "Natalie",
    "Raelynn",
    "Sophie",
    "Maria",
    "Aubrey",
    "Amara",
    "Zoe",
    "Josephine",
    "Kehlani",
    "Madison",
    "Emery",
    "Gabriella",
    "Hannah",
    "Addison",
    "Adeline",
    "Bella",
    "Oakley",
    "Ruby",
    "Sadie",
    "Eden",
    "Everly",
    "Audrey",
    "Clara",
    "Alice",
    "Autumn",
    "Lyla",
    "Leah",
    "Lillian",
    "Millie",
    "Oaklynn",
    "Kennedy",
    "Madelyn",
    "Maeve",
    "Skylar",
    "Amira",
    "Claire",
    "Daisy",
    "Savannah",
    "Anna",
    "Genesis",
    "Vivian",
    "Noah",
    "Liam",
    "Oliver",
    "Elijah",
    "Lucas",
    "Mateo",
    "Ezra",
    "Levi",
    "Asher",
    "Luca",
    "Michael",
    "James",
    "Henry",
    "Maverick",
    "Hudson",
    "Ethan",
    "Leo",
    "Gabriel",
    "Elias",
    "Sebastian",
    "Grayson",
    "Wyatt",
    "Jack",
    "Theodore",
    "William",
    "Mason",
    "Muhammad",
    "Benjamin",
    "Samuel",
    "Alexander",
    "Daniel",
    "Theo",
    "Isaiah",
    "Aiden",
    "Owen",
    "John",
    "Kai",
    "Josiah",
    "David",
    "Jackson",
    "Ezekiel",
    "Waylon",
    "Anthony",
    "Carter",
    "Luke",
    "Jayden",
    "Santiago",
    "Cooper",
    "Eli",
    "Julian",
    "Isaac",
    "Silas",
    "Matthew",
    "Micah",
    "Nathan",
    "Logan",
    "Atlas",
    "Weston",
    "Roman",
    "Miles",
    "Caleb",
    "Lincoln",
    "Amir",
    "Adam",
    "Jace",
    "Joseph",
    "Rowan",
    "Jacob",
    "Joshua",
    "Enzo",
    "Thomas",
    "Beau",
    "Nolan",
    "Jaxon",
    "Jeremiah",
    "Christian",
    "Parker",
    "Christopher",
    "Adrian",
    "Walker",
    "Wesley",
    "Luka",
    "Matteo",
    "Zion",
    "Axel",
    "Landon",
    "Easton",
    "Greyson",
    "Amari",
    "Colton",
    "Charlie",
    "Kayden",
    "Xavier",
    "Adriel",
    "Legend",
    "Malachi",
    "Dylan",
    "River",
    "Andrew",
    "Dominic",
]


@dataclass
class Customer:
    id: int
    name: str
    region: str
    birth_date: date
    joined_date: date

    def as_list(self):
        return [self.id, self.name, self.region, self.birth_date, self.joined_date]


@dataclass
class Transaction:
    customer_id: int
    date: date
    payment_type: str
    total: float
    _id: Optional[int] = None

    def get_id(self) -> Optional[int]:
        return self._id

    def set_id(self, id: int):
        self._id = id

    def as_list(self):
        return [self._id, self.customer_id, self.date, self.payment_type, self.total]


def write_csv(file_path, data):
    with open(file_path, "w", newline="") as csvfile:
        file = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for row in data:
            file.writerow(row.as_list())


def random_date(start=date(1980, 1, 1), end=date(2000, 1, 1)):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    return start + timedelta(random.randrange(delta.days))


def random_region():
    return random.choice(["NA", "EU", "APAC", "LATAM"])


customers: List[Customer] = []
joined_date = date(2023, 1, 1)
random.shuffle(names)  # shuffle in place
for id, name in enumerate(names):
    while random.randint(0, 10) > 5:
        joined_date += timedelta(days=1)
    customers.append(
        Customer(
            id=id,
            name=name,
            region=random_region(),
            birth_date=random_date(),
            joined_date=joined_date,
        )
    )

write_csv(CUSTOMERS_FILE, customers)


transactions: List[Transaction] = []
last_transaction_date = date(2024, 1, 1)
for id in range(1000):
    random_customer = customers[random.randint(0, len(customers) - 1)]
    transactions.append(
        Transaction(
            customer_id=random_customer.id,
            date=random_date(start=random_customer.joined_date, end=last_transaction_date),
            payment_type=random.choice(["credit", "debit", "cash"]),
            total=random.randint(100, 2000) / 100.0,
        )
    )
for id, transaction in enumerate(sorted(transactions, key=lambda x: x.date)):
    transaction.set_id(id=id)


write_csv(TRANSACTIONS_FILE, sorted(transactions, key=lambda x: x.get_id() or 0))
