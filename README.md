# 📦 IntelliSupply: Advanced Supply Chain & Pricing Optimization System

A **data-driven platform** for optimizing supply chain operations with **14+ predictive models**.  
Handles **demand forecasting**, **dynamic pricing**, **inventory risk analysis**, and **supplier performance scoring** — all powered by a **FastAPI backend** for **real-time predictions** and **smooth system integration**.

---

## 🌟 Overview
**IntelliSupply** is a **comprehensive supply chain optimization platform** built to solve challenges in **pricing, demand, inventory, and supplier management**.  
It integrates **14 trained predictive models** with a **FastAPI backend** for **fast, accurate, and scalable decision-making**.

---

## ✨ Features
- 📈 **Demand Forecasting** – Predict future product demand to avoid over/under-stocking.
- 💰 **Dynamic Pricing** – Optimize prices using competition, seasonality, and profit margins.
- 📦 **Inventory Management** – Reorder triggers, stockout prediction, and expiry risk detection.
- 🏭 **Supplier Analytics** – Evaluate suppliers based on delay, quality, and cost metrics.
- ⏱️ **Order & Delivery Delay Prediction** – Anticipate bottlenecks for better planning.
- 🧩 **Anomaly Detection** – Detect irregularities in supply chain data for proactive fixes.
- ⚡ **FastAPI Integration** – REST API endpoints for seamless integration into any system.

---

## 🛠 Tech Stack
- **Python** – Core implementation and orchestration
- **FastAPI** – ⚡ High-performance backend framework
- **scikit-learn** – 🧠 Predictive modeling & machine learning
- **pandas / NumPy** – 📊 Data preprocessing and manipulation
- **Pickle** – 🔐 Model serialization for quick deployment
- **ETL Pipelines** – Efficient data handling for sales, pricing, inventory, and supplier datasets

---

## 📂 Project Structure
```bash
IntelliSupply/
├── models/                 # Pre-trained predictive models (.pkl)
├── datasets/               # Training & validation datasets
├── main.py                 # FastAPI server entry point
├── model_preparation/      # Jupyter notebooks for model training
├── test_api_endpoints.py   # API testing and validation scripts
└── requirements.txt        # Project dependencies
