from typing import List, Type, Dict, Literal, Union, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import websockets
import asyncio
from sqlalchemy import Column, Integer, Float, String, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Query
from contextlib import contextmanager
import hashlib
import json

# ----------------------
# ORM Base Configuration
# ----------------------
Base = declarative_base()
engine = create_engine('postgresql://user:pass@localhost/football')
Session = sessionmaker(bind=engine)

@contextmanager
def db_session():
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

# ----------------------
# Data Version Control
# ----------------------
class DataVersion(BaseModel):
    hash: str
    timestamp: datetime
    previous: Optional[str]
    message: str

def version_control(func):
    def wrapper(self, *args, **kwargs):
        prev_hash = self.version.hash if self.version else ""
        new_version = DataVersion(
            hash=hashlib.sha256(self.json().encode()).hexdigest(),
            timestamp=datetime.now(),
            previous=prev_hash,
            message=f"Updated by {func.__name__}"
        )
        self.version = new_version
        return func(self, *args, **kwargs)
    return wrapper

# ----------------------
# Position Classes
# ----------------------
class Position(str, Enum):
    GOALKEEPER = "GK"
    DEFENDER = "DF"
    MIDFIELDER = "MF"
    FORWARD = "FW"
    WINGER = "WG"

class TrackingData(BaseModel):
    timestamp: datetime
    coordinates: Dict[Literal["x", "y", "z"], float]
    speed: float
    heart_rate: int

class PositionSpecificMetrics(Base):
    __tablename__ = "position_metrics"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer)
    position = Column(String)
    metrics = Column(JSON)

# ----------------------
# Base Player Model
# ----------------------
class PlayerBase(BaseModel):
    player_id: int = Field(..., gt=0)
    team_id: int = Field(..., gt=0)
    position: Position
    version: Optional[DataVersion]
    tracking_data: List[TrackingData] = []

    @validator('position')
    def validate_position(cls, v):
        if v == Position.GOALKEEPER and 'GoalkeeperData' not in cls.__name__:
            raise ValueError("Invalid position for player type")
        return v

    # ----------------------
    # Real-Time Updates
    # ----------------------
    async def real_time_updates(self, uri: str):
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                update = json.loads(message)
                if update['player_id'] == self.player_id:
                    self.apply_update(update)

    def apply_update(self, update: Dict):
        self.tracking_data.append(TrackingData(**update))
        self.calculate_injury_risk()

    # ----------------------
    # Injury Risk Assessment
    # ----------------------
    def calculate_injury_risk(self, window: int = 5) -> float:
        recent = self.tracking_data[-window:]
        load = sum(td.speed**2 * td.heart_rate for td in recent)
        return min(load / 1e6, 1.0)

    # ----------------------
    # Spatial Visualization
    # ----------------------
    def plot_heatmap(self):
        df = pd.DataFrame([td.coordinates for td in self.tracking_data])
        plt.hexbin(df['x'], df['y'], gridsize=20, cmap='Reds')
        plt.title(f"Position Heatmap for Player {self.player_id}")
        plt.savefig(f"heatmap_{self.player_id}.png")
        plt.close()

    # ----------------------
    # Performance Benchmarks
    # ----------------------
    def compare_benchmark(self, benchmark: Dict) -> Dict:
        return {
            'speed': self.avg_speed() / benchmark['speed'],
            'distance': self.total_distance() / benchmark['distance'],
            'intensity': self.match_intensity() / benchmark['intensity']
        }

    def avg_speed(self) -> float:
        return np.mean([td.speed for td in self.tracking_data])

    def total_distance(self) -> float:
        return sum(np.linalg.norm(
            [td.coordinates['x'], td.coordinates['y']]
        ) for td in self.tracking_data)

    def match_intensity(self) -> float:
        return np.mean([td.heart_rate for td in self.tracking_data])

# ----------------------
# Position-Specific Subclasses
# ----------------------
class GoalkeeperData(PlayerBase):
    saves: int = 0
    claims: int = 0
    punches: int = 0
    goals_conceded: int = 0

    class ORM(Base):
        __tablename__ = "goalkeepers"
        id = Column(Integer, primary_key=True)
        player_id = Column(Integer)
        saves = Column(Integer)
        claims = Column(Integer)
        punches = Column(Integer)
        goals_conceded = Column(Integer)

class DefenderData(PlayerBase):
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    aerial_duels_won: int = 0

class MidfielderData(PlayerBase):
    key_passes: int = 0
    pass_accuracy: float = 0.0
    final_third_entries: int = 0

class ForwardData(PlayerBase):
    shots_on_target: int = 0
    dribbles: int = 0
    expected_goals: float = 0.0

# ----------------------
# Database Integration
# ----------------------
class PlayerManager:
    # Registry mapping player classes to their ORM counterparts
    _registry: Dict[Type[PlayerBase], Type[Base]] = {
        GoalkeeperData: GoalkeeperData.ORM,
        DefenderData: DefenderData.ORM,
        MidfielderData: MidfielderData.ORM,
        ForwardData: ForwardData.ORM,
        WingerData: WingerData.ORM
    }

    # Reverse mapping for ORM to player class resolution
    _orm_to_player: Dict[Type[Base], Type[PlayerBase]] = {
        orm_class: player_class 
        for player_class, orm_class in _registry.items()
    }

    @classmethod
    def save_to_db(cls, player: PlayerBase) -> None:
        """Save player data to the appropriate position-specific table"""
        with db_session() as session:
            orm_class = cls._registry.get(type(player))
            if not orm_class:
                raise ValueError(f"Unsupported player type: {type(player).__name__}")
            
            # Convert player data to ORM model
            orm_data = {
                col.name: getattr(player, col.name)
                for col in orm_class.__table__.columns
                if hasattr(player, col.name)
            }
            orm_instance = orm_class(**orm_data)
            
            # Update existing record or create new one
            existing = session.query(orm_class).filter_by(player_id=player.player_id).first()
            if existing:
                for key, value in orm_data.items():
                    setattr(existing, key, value)
            else:
                session.add(orm_instance)
                
            session.commit()

    @classmethod
    def load_from_db(cls, player_id: int) -> PlayerBase:
        """Load player data from the database with position detection"""
        with db_session() as session:
            for orm_class, player_class in cls._orm_to_player.items():
                result: Base = session.query(orm_class).filter_by(player_id=player_id).first()
                if result:
                    return cls._orm_to_player_dict(result, player_class)
            raise ValueError(f"Player {player_id} not found in any position table")

    @classmethod
    def _orm_to_player_dict(cls, orm_instance: Base, player_class: Type[PlayerBase]) -> PlayerBase:
        """Convert SQLAlchemy ORM instance to Pydantic model"""
        return player_class(**{
            col.name: getattr(orm_instance, col.name)
            for col in orm_instance.__table__.columns
        })

    @classmethod
    def query_players(cls, position: Type[PlayerBase] = None, **filters) -> List[PlayerBase]:
        """Generic query interface for players"""
        with db_session() as session:
            query = session.query
            if position:
                orm_class = cls._registry[position]
                query = query(orm_class)
                if filters:
                    query = query.filter_by(**filters)
                return [cls._orm_to_player_dict(instance, position) 
                        for instance in query.all()]
            
            # Search across all position tables if no specific position given
            results = []
            for orm_class, player_class in cls._orm_to_player.items():
                query = session.query(orm_class)
                if filters:
                    query = query.filter_by(**filters)
                results.extend([
                    cls._orm_to_player_dict(instance, player_class)
                    for instance in query.all()
                ])
            return results
