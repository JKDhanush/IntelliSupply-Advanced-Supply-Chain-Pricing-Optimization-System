from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# -------- Load all models --------
m1 = joblib.load("models/demand_forecast_model.pkl")
m2 = joblib.load("models/dynamic_pricing_model.pkl")
m3 = joblib.load("models/reorder_level_model.pkl")
m4 = joblib.load("models/stockout_model.pkl")
m5 = joblib.load("models/supplier_delay_model.pkl")
m6 = joblib.load("models/profit_margin_model.pkl")
m7 = joblib.load("models/calendar_holiday_model.pkl")
m8 = joblib.load("models/order_delay_model.pkl")

# -------- Define input schemas (Enhanced Version) --------
class DemandInput(BaseModel):
    price: float
    discount: float
    day_of_week: int
    month: int
    is_weekend: int
    is_festival: int
    season_encoded: int
    category_encoded: int
    shelf_life_days: int

class PricingInput(BaseModel):
    discount: float
    competitor_price: float
    profit_margin: float
    cost_price: float
    category_encoded: int
    preferred: int

class ReorderInput(BaseModel):
    current_stock: int
    days_to_expire: int
    shelf_life_days: int
    return_rate: float
    category_encoded: int

class StockoutInput(BaseModel):
    units_sold: int
    discount: float
    day_number: int
    is_festival: int
    category_encoded: int
    preferred: int
    reorder_level: int

class SupplierInput(BaseModel):
    default_lead_days: int
    avg_delivery_delay: float
    quality_score: float
    return_rate: float
    preferred: int
    supply_cost_index: float

class ProfitInput(BaseModel):
    price: float
    competitor_price: float
    discount: float
    quality_score: float
    category_encoded: int

class HolidayInput(BaseModel):
    day_number: int
    is_weekend: int
    is_festival: int
    season_encoded: int
    week_of_year: int

class OrderInput(BaseModel):
    quantity_ordered: int
    avg_delivery_delay: float
    quality_score: float
    category_encoded: int

# -------- API Endpoints --------
@app.post("/predict/demand")
def predict_demand(inp: DemandInput):
    X = [[inp.price, inp.discount, inp.day_of_week, inp.month, inp.is_weekend, inp.is_festival, inp.season_encoded, inp.category_encoded, inp.shelf_life_days]]
    return {"units_sold": m1.predict(X)[0]}

@app.post("/predict/price")
def predict_price(inp: PricingInput):
    X = [[inp.discount, inp.competitor_price, inp.profit_margin, inp.cost_price, inp.category_encoded, inp.preferred]]
    return {"price": m2.predict(X)[0]}

@app.post("/predict/reorder")
def predict_reorder(inp: ReorderInput):
    X = [[inp.current_stock, inp.days_to_expire, inp.shelf_life_days, inp.return_rate, inp.category_encoded]]
    return {"reorder_level": m3.predict(X)[0]}

@app.post("/predict/stockout")
def predict_stockout(inp: StockoutInput):
    X = [[inp.units_sold, inp.discount, inp.day_number, inp.is_festival, inp.category_encoded, inp.preferred, inp.reorder_level]]
    return {"stockout_risk": int(m4.predict(X)[0])}

@app.post("/predict/supplier_delay")
def predict_supplier_delay(inp: SupplierInput):
    X = [[inp.default_lead_days, inp.avg_delivery_delay, inp.quality_score, inp.return_rate, inp.preferred, inp.supply_cost_index]]
    return {"on_time_delivery_rate": m5.predict(X)[0]}

@app.post("/predict/profit")
def predict_profit(inp: ProfitInput):
    X = [[inp.price, inp.competitor_price, inp.discount, inp.quality_score, inp.category_encoded]]
    return {"profit_margin": m6.predict(X)[0]}

@app.post("/predict/holiday")
def predict_holiday(inp: HolidayInput):
    X = [[inp.day_number, inp.is_weekend, inp.is_festival, inp.season_encoded, inp.week_of_year]]
    return {"is_holiday": int(m7.predict(X)[0])}

@app.post("/predict/order_delay")
def predict_order_delay(inp: OrderInput):
    X = [[inp.quantity_ordered, inp.avg_delivery_delay, inp.quality_score, inp.category_encoded]]
    return {"delay_days": m8.predict(X)[0]}

print("🚀 FastAPI ML prediction service is ready!")