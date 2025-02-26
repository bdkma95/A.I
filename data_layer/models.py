from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, List, Literal
from datetime import datetime, timezone
from enum import Enum
import pandas as pd
import numpy as np

class Position(str, Enum):
    GOALKEEPER = "GK"
    DEFENDER = "DF"
    MIDFIELDER = "MF"
    FORWARD = "FW"

class PlayerData(BaseModel):
    # Core Identification
    player_id: int = Field(..., gt=0, description="Unique player identifier")
    team_id: int = Field(..., gt=0, description="Team identifier")
    match_id: int = Field(..., gt=0, description="Match identifier")
    
    # Basic Performance Metrics
    goals: int = Field(0, ge=0, description="Goals scored")
    assists: int = Field(0, ge=0, description="Assists provided")
    minutes_played: int = Field(..., ge=0, le=120, description="Minutes played (0-120)")
    
    # Advanced Metrics
    expected_goals: float = Field(0.0, ge=0, description="xG value")
    expected_assists: float = Field(0.0, ge=0, description="xA value")
    pass_accuracy: float = Field(..., ge=0, le=1.0, description="Pass completion percentage")
    
    # Physical Metrics
    distance_covered: float = Field(..., ge=0, description="Meters covered")
    top_speed: float = Field(..., ge=0, le=12.0, description="Top speed in m/s")
    sprints: int = Field(0, ge=0, description="Number of sprints")
    
    # Positional Data
    position: Position
    position_heatmap: Dict[str, float] = Field(
        default_factory=dict, 
        description="Heatmap coordinates frequency"
    )
    
    # Temporal Data
    match_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of match"
    )
    
    # Validation Rules
    @validator("goals", "assists", pre=True)
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v < 0:
            raise ValueError("Must be a non-negative integer")
        return v
    
    @root_validator
    def validate_minutes_consistency(cls, values):
        mins = values.get("minutes_played", 0)
        if mins == 0 and (values["goals"] > 0 or values["assists"] > 0):
            raise ValueError("Player with 0 minutes cannot have goals/assists")
        return values

    # Derived Properties
    @property
    def goal_contribution(self) -> float:
        """Total direct goal contributions"""
        return self.goals + self.assists
    
    @property
    def efficiency_ratio(self) -> float:
        """Minutes per goal contribution"""
        if self.minutes_played == 0:
            return 0.0
        return self.goal_contribution / self.minutes_played
    
    # Serialization Methods
    def to_feature_dict(self) -> Dict:
        """Format for ML model input"""
        return {
            "player_id": self.player_id,
            "xg": self.expected_goals,
            "xa": self.expected_assists,
            "speed": self.top_speed,
            "pass_acc": self.pass_accuracy,
            "position": self.position.value
        }

    def to_parquet(self, path: str) -> None:
        """Export to Parquet format"""
        pd.DataFrame([self.dict()]).to_parquet(path)

    # Analytics Methods
    def fitness_score(self, weights: Dict = None) -> float:
        """Calculate physical fitness score"""
        default_weights = {
            "distance_covered": 0.4,
            "top_speed": 0.3,
            "sprints": 0.3
        }
        weights = weights or default_weights
        
        return (
            weights["distance_covered"] * np.log1p(self.distance_covered) +
            weights["top_speed"] * self.top_speed +
            weights["sprints"] * self.sprints
        )

    # Metadata
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            Position: lambda pos: pos.value
        }
        schema_extra = {
            "example": {
                "player_id": 101,
                "team_id": 202,
                "match_id": 303,
                "goals": 2,
                "assists": 1,
                "minutes_played": 90,
                "expected_goals": 1.8,
                "expected_assists": 0.9,
                "pass_accuracy": 0.85,
                "distance_covered": 10850.5,
                "top_speed": 10.2,
                "sprints": 25,
                "position": "FW",
                "position_heatmap": {"x": 0.7, "y": 0.3},
                "match_date": "2023-08-15T19:30:00+00:00"
            }
        }

class GoalkeeperData(PlayerData):
    # Specialized goalkeeper metrics
    saves: int = Field(0, ge=0)
    claims: int = Field(0, ge=0)
    punches: int = Field(0, ge=0)
    goals_conceded: int = Field(0, ge=0)
    
    @validator("position")
    def validate_position(cls, v):
        if v != Position.GOALKEEPER:
            raise ValueError("Position must be GK for goalkeepers")
        return v

class MatchPerformance:
    """Container for multiple player performances"""
    def __init__(self, performances: List[PlayerData]):
        self.players = performances
        
    def team_summary(self, team_id: int) -> Dict:
        """Generate team-level summary statistics"""
        team_players = [p for p in self.players if p.team_id == team_id]
        return {
            "total_goals": sum(p.goals for p in team_players),
            "total_xg": sum(p.expected_goals for p in team_players),
            "avg_distance": np.mean([p.distance_covered for p in team_players])
        }
