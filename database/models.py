import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, BigInteger, DateTime, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'finance')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'finance123')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'finance_ml')}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class RawPrice(Base):
    __tablename__ = "raw_prices"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

    __table_args__ = (UniqueConstraint("ticker", "timestamp"),)


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    return_1 = Column(Float)
    return_5 = Column(Float)
    return_15 = Column(Float)
    vol_5 = Column(Float)
    vol_15 = Column(Float)
    volume_ratio = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_position = Column(Float)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    label = Column(Integer)

    __table_args__ = (UniqueConstraint("ticker", "timestamp"),)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    model_version = Column(String(50))
    predicted_label = Column(Integer)
    probability = Column(Float)
    actual_label = Column(Integer)

    __table_args__ = (UniqueConstraint("ticker", "timestamp"),)
