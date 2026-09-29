"""
Pydantic models: they validate the request body and document the response.

The field names match the columns of the Kaggle dataset, so the same customer
record can be sent to the API without any renaming.
"""

from pydantic import BaseModel, ConfigDict, Field

# Repayment status: -2 = no consumption, -1 = paid duly, 0 = revolving credit,
# 1..9 = number of months of payment delay.
PAY_STATUS_DESCRIPTION = "Repayment status: -2/-1/0 = on time or no use, 1-9 = months of delay"


class CustomerFeatures(BaseModel):
    """Customer attributes used by the model (amounts in NT dollars)."""

    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit")
    SEX: int = Field(..., ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(
        ..., ge=0, le=6,
        description="1 = graduate school, 2 = university, 3 = high school, 4 = others (0, 5, 6 are grouped into 4)",
    )
    MARRIAGE: int = Field(
        ..., ge=0, le=3, description="1 = married, 2 = single, 3 = others (0 is grouped into 3)"
    )
    AGE: int = Field(..., ge=18, le=100, description="Age in years")

    PAY_0: int = Field(..., ge=-2, le=9, description=f"September. {PAY_STATUS_DESCRIPTION}")
    PAY_2: int = Field(..., ge=-2, le=9, description=f"August. {PAY_STATUS_DESCRIPTION}")
    PAY_3: int = Field(..., ge=-2, le=9, description=f"July. {PAY_STATUS_DESCRIPTION}")
    PAY_4: int = Field(..., ge=-2, le=9, description=f"June. {PAY_STATUS_DESCRIPTION}")
    PAY_5: int = Field(..., ge=-2, le=9, description=f"May. {PAY_STATUS_DESCRIPTION}")
    PAY_6: int = Field(..., ge=-2, le=9, description=f"April. {PAY_STATUS_DESCRIPTION}")

    # Bill amounts can be negative (customer overpaid).
    BILL_AMT1: float = Field(..., description="Bill statement, September")
    BILL_AMT2: float = Field(..., description="Bill statement, August")
    BILL_AMT3: float = Field(..., description="Bill statement, July")
    BILL_AMT4: float = Field(..., description="Bill statement, June")
    BILL_AMT5: float = Field(..., description="Bill statement, May")
    BILL_AMT6: float = Field(..., description="Bill statement, April")

    PAY_AMT1: float = Field(..., ge=0, description="Amount paid, September")
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid, August")
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid, July")
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid, June")
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid, May")
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid, April")

    # This example pre-fills the request in Swagger (/docs).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 50000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 35,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 46000,
                "BILL_AMT2": 45000,
                "BILL_AMT3": 43000,
                "BILL_AMT4": 30000,
                "BILL_AMT5": 29000,
                "BILL_AMT6": 28000,
                "PAY_AMT1": 0,
                "PAY_AMT2": 1500,
                "PAY_AMT3": 1200,
                "PAY_AMT4": 1100,
                "PAY_AMT5": 1000,
                "PAY_AMT6": 1000,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = high default risk, 0 = low default risk")
    label: str = Field(..., description="high_default_risk or low_default_risk")
    default_probability: float = Field(..., ge=0, le=1)
    threshold: float
    model_version: str
    model_uri: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
