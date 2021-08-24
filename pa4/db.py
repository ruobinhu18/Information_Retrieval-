from pathlib import Path
from typing import Dict, Any

from peewee import SqliteDatabase, Model  # type: ignore
from peewee import IntegerField  # type: ignore
from utils import load_wapo


# create a sqlite database stored in pa4_data/wapo_docs.db
db = SqliteDatabase(
    Path(__file__).parent.joinpath("pa4_data/wapo_docs.db"),
    pragmas={
        "journal_mode": "wal",
        "cache_size": -1 * 64000,  # 64 Mb
        "foreign_keys": 1,
        "ignore_check_constraints": 0,
        "synchronous": 0,
    },
)


class BaseModel(Model):
    class Meta:
        database = db


class Doc(BaseModel):
    """
    define your WAPO doc data model (table schema)
    reference: http://docs.peewee-orm.com/en/latest/peewee/models.html#
    """

    doc_id = IntegerField(primary_key=True)
    # TODO: add more fields


@db.connection_context()
def create_tables():
    """
    create and populate the wapo doc table. Consider using bulk insert to load data faster
    reference: http://docs.peewee-orm.com/en/latest/peewee/querying.html#bulk-inserts
    :return:
    """
    # TODO:


@db.connection_context()
def query_doc(doc_id: int) -> Dict[str, Any]:
    """
     given the doc_id, get the document dict
     reference:
     http://docs.peewee-orm.com/en/latest/peewee/querying.html#selecting-a-single-record
     http://docs.peewee-orm.com/en/latest/peewee/playhouse.html#model_to_dict
    :param doc_id:
    :return:
    """
    # TODO:


if __name__ == "__main__":
    create_tables()
