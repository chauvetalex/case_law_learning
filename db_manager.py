# TODO en plus de créer un table pour les réponses alternative, créer un table permettant de décomposer les réponses en points.

from typing import Optional, List
import pathlib
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

DB = config.get('PATHS', 'DB')

from sqlalchemy import ForeignKey, String, Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy import insert, update, select, delete

class Base(DeclarativeBase):
    pass

# Créer une table contenant les arrêts de jurisprudence.
class QASource(Base):
    __tablename__ = "tbl_qa_sources"

    id: Mapped[str] = mapped_column(primary_key=True)
    content: Mapped[str]
    raw_text: Mapped[Optional[str]]
    level: Mapped[int]

    # alt_answers: Mapped[List["AltAnswer"]] = relationship(back_populates='qa_source')

# Créer une table permettant d'ajouter des réponses alternatives.
class AltAnswer(Base):
    __tablename__ = "tbl_alt_answers"

    text: Mapped[str] = mapped_column(primary_key=True)
    # foreign_id: Mapped[str] = mapped_column(ForeignKey("tbl_qa_sources.id"))
    foreign_id: Mapped[str]

    # qa_source: Mapped["QASource"] = relationship(back_populates='alt_answers')

# ! Les foreign keys sont mal gérée par sqlite3.
# Le PRAGMA doit être rappelé à chaque connection à la base de données.
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Créer les tables, engine et session.
engine = create_engine(f'sqlite:///{DB}')
if pathlib.Path(DB).exists() is False:
    Base.metadata.create_all(engine)
session = Session(engine)

def get_records_as_dict():
    sel_sql = select(QASource.id, QASource.content, QASource.raw_text, QASource.level).where(QASource.content != None)
    records = session.execute(sel_sql).all()
    records = [rec._asdict() for rec in records]

    return records

def add_new_answer(foreign_id, new_answer):
    insert_sql = insert(AltAnswer).values({'foreign_id':foreign_id, 'text':new_answer})
    session.execute(insert_sql)
    session.commit()

# TODO
def get_alternative_answers():
    pass
